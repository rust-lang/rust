//@ compile-flags: -Zunpretty=ast-tree
#![c={#![c[)x   //~ ERROR mismatched closing delimiter
    //~ ERROR this file contains an unclosed delimiter
