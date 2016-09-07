#![feature(custom_attribute, attr_literals)]
#![miri(stack_limit=2)]

fn bar() {
    foo();
}

fn foo() {
    cake(); //~ ERROR reached the configured maximum number of stack frames
}

fn cake() {
    flubber();
}

fn flubber() {}

fn main() {
    bar();
}
