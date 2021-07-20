// check-pass
#![allow(unused, dead_code)]

fn foo() -> u32 {
    return 'label: loop { break 'label 42; };
}

fn bar() -> u32 {
    loop { break 'label: loop { break 'label 42; }; }
}

pub fn main() {
    // Regression test for issue #86948:
    let a = 'first_loop: loop {
        break 'first_loop 1;
    };
    let b = loop {
        break 'inner_loop: loop {
            break 'inner_loop 1;
        };
    };
}
