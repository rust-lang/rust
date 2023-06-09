use std::sync::Mutex;

static GLOBAL: Mutex<&'static str> = Mutex::new("global str");

struct Foo<T>(T); // `T` is covariant.

fn foo() {
    let mut x: Foo<for<'a> fn(&'a str)> = Foo(|_| ());
    let Foo(ref mut y): Foo<fn(&'static str)> = x;
    //~^ ERROR mismatched types
    *y = |s| *GLOBAL.lock().unwrap() = s;
    let string = String::from("i am shortlived");
    (x.0)(&string);
}

fn main() {
    foo();
    println!("{}", GLOBAL.lock().unwrap());
}
