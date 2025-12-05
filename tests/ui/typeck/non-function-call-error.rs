//! Regression test for issue #10969, #22468

fn main() {
    let foo = "bar";
    let x = foo("baz");
    //~^ ERROR: expected function, found `&str`

    let i = 0i32;
    i();
    //~^ ERROR expected function, found `i32`
}

fn foo(file: &str) -> bool {
    true
}

fn func(i: i32) {
    i(); //~ERROR expected function, found `i32`
}
