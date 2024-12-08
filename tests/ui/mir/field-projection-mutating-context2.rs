use std::sync::Mutex;

static GLOBAL: Mutex<&'static str> = Mutex::new("global str");

struct Foo<T>(T); // `T` is covariant.

fn foo<'a>(mut x: Foo<fn(&'a str)>, string: &'a str) {
    let Foo(ref mut y): Foo<fn(&'static str)> = x;
    //~^ ERROR lifetime may not live long enough
    *y = |s| *GLOBAL.lock().unwrap() = s;
    (x.0)(&string);
}

fn main() {
    foo(Foo(|_| ()), &String::from("i am shortlived"));
    println!("{}", GLOBAL.lock().unwrap());
}
