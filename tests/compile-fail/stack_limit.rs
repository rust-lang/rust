#![feature(custom_attribute, attr_literals)]
#![miri(stack_limit=16)]

fn bar() {
    foo();
}

fn foo() {
    cake(); //~ ERROR reached the configured maximum number of stack frames
}

fn cake() {
    flubber(3);
}

fn flubber(i: u32) {
    if i > 0 {
        flubber(i-1);
    } else {
        bar();
    }
}

fn main() {
    bar();
}
