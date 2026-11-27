// Regression test for https://github.com/rust-lang/rust/issues/166272

//@ edition:2021

fn first_in_block() {
    || Box::new(42) as Box::<dyn _>;
    //~^ ERROR expected a trait, found type
}

fn later_in_block() {
    let _x = 1;
    || Box::new(42) as Box::<dyn _>;
    //~^ ERROR expected a trait, found type
}

fn in_let() {
    let _x = || Box::new(42) as Box::<dyn _>;
    //~^ ERROR expected a trait, found type
}

fn in_call() {
    let _f = foo;
    foo(|| Box::new(42) as Box::<dyn _>);
    //~^ ERROR expected a trait, found type
}

fn foo(_: impl Fn()) {}

fn main() {}
