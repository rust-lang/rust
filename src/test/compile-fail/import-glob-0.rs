// error-pattern: unresolved name

import module_of_many_things::*;

mod module_of_many_things {
    export f1;
    export f2;
    export f4;

    fn f1() { log "f1"; }
    fn f2() { log "f2"; }
    fn f3() { log "f3"; }
    fn f4() { log "f4"; }
}


fn main() {
    f1();
    f2();
    f999(); // 'export' currently doesn't work?
    f4();
}
