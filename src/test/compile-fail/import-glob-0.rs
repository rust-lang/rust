// error-pattern: unresolved name

import module_of_many_things::*;

mod module_of_many_things {
    export f1;
    export f2;
    export f4;

    fn f1() { #debug("f1"); }
    fn f2() { #debug("f2"); }
    fn f3() { #debug("f3"); }
    fn f4() { #debug("f4"); }
}


fn main() {
    f1();
    f2();
    f999(); // 'export' currently doesn't work?
    f4();
}
