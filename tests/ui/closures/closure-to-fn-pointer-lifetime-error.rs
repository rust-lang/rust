//! Regression test for https://github.com/rust-lang/rust/issues/40000

fn main() {
    let bar: fn(&mut u32) = |_| {};

    fn foo(x: Box<dyn Fn(&i32)>) {}
    let bar = Box::new(|x: &i32| {}) as Box<dyn Fn(_)>;
    foo(bar);
    //~^ ERROR higher-ranked lifetime error: lifetime outlives placeholder
    //~| ERROR higher-ranked lifetime error: lifetime outlives placeholder
}
