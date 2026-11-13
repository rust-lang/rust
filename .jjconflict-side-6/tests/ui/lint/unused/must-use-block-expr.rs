//@ run-rustfix
//@ check-pass

#![warn(unused_must_use)]

#[must_use]
fn foo() -> i32 {
    42
}

fn bar() {
    {
        foo();
        //~^ WARN unused return value
    }
}

fn baz() {
    {
        foo()
        //~^ WARN unused return value
    };
}

fn main() {
    bar();
    baz();
    {
        1 + 2;
        //~^ WARN unused arithmetic operation
    }
    {
        1 + 2
        //~^ WARN unused arithmetic operation
    };
}
