# The Hors Scrapper

The script can be used to recursively download and order structures from https://www.vergiliusproject.com/
It will download all required symbols and put it into a C++ header file.

Usage:
    python vergilius_scraper.py <url> [output_file.hpp]
    
Example:
    python vergilius_scraper.py https://www.vergiliusproject.com/kernels/x64/windows-11/24h2/_KPROCESS output.hpp

It is async because a lot of headers have a bunch of dependencies, however it's not meant to be the fastest so vergilius can cope with it.

# The LSP scrapper

There is an optional hpp_analyzer.hpp utility which uses the vergilius scrapper, but also utilizes LSP to re-order structures and fix compilation errors, if clangd is found.
It can also work on existing header monoliths to prevent duplicate resolving and save precious internet bandwidth.


### Finally, you can put on the iron hooves and paste faster!
