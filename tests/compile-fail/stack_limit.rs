#![feature(custom_attribute, attr_literals)]
#![miri(stack_limit=16)]

//error-pattern: reached the configured maximum number of stack frames

fn bar() {
    foo();
}

fn foo() {
    cake();
}

fn cake() {
    bar();
}

fn main() {
    bar();
}
