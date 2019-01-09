fn main() {
    let bar: fn(&mut u32) = |_| {};

    fn foo(x: Box<Fn(&i32)>) {}
    let bar = Box::new(|x: &i32| {}) as Box<Fn(_)>;
    foo(bar); //~ ERROR E0308
}
