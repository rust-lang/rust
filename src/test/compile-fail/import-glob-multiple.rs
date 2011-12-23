// error-pattern:common2

import mod1::*;
import mod2::*;

mod mod1 {
    fn f1() { #debug("f1"); }
    fn common1() { #debug("common") }
    fn common2() { #debug("common") }
}

mod mod2 {
    fn f2() { #debug("f1"); }
    fn common1() { #debug("common") }
    fn common2() { #debug("common") }
}



fn main() { common2(); }
