// compile-pass
// skip-codegen
#![allow(warnings)]

enum E {
    A = {
        enum F { B }
        0
    }
}


fn main() {}
