// Issue #787
// Don't try to clean up uninitizaed locals

use std;

fn test_break() {
    while true {
        let x: @int = break;
    }
}

fn test_cont() {
    let i = 0;
    while i < 1 {
        i += 1;
        let x: @int = cont;
    }
}

fn test_ret() {
    let x: @int = ret;
}

fn test_fail() {
    fn f() {
        std::task::unsupervise();
        let x: @int = fail;
    }
    let g = f;
    std::task::spawn(g);
}

fn test_fail_indirect() {
    fn f() -> ! {
        fail;
    }
    fn g() {
        std::task::unsupervise();
        let x: @int = f();
    }
    let h = g;
    std::task::spawn(h);
}

fn main() {
    test_break();
    test_cont();
    test_ret();
    test_fail();
    test_fail_indirect();
}