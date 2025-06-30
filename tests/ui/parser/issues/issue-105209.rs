//@ compile-flags: -Zunpretty=ast-tree
#![c={#![c[)x
                //~ ERROR this file contains an unclosed delimiter
