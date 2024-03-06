//@ run-pass
#![allow(unused_variables)]
/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

fn call_it<F>(f: F)
    where F : FnOnce(String) -> String
{
    println!("{}", f("Fred".to_string()))
}

fn call_a_thunk<F>(f: F) where F: FnOnce() {
    f();
}

fn call_this<F>(f: F) where F: FnOnce(&str) + Send {
    f("Hello!");
}

fn call_bare(f: fn(&str)) {
    f("Hello world!")
}

fn call_bare_again(f: extern "Rust" fn(&str)) {
    f("Goodbye world!")
}

pub fn main() {
    // Procs

    let greeting = "Hello ".to_string();
    call_it(|s| {
        format!("{}{}", greeting, s)
    });

    let greeting = "Goodbye ".to_string();
    call_it(|s| format!("{}{}", greeting, s));

    let greeting = "How's life, ".to_string();
    call_it(|s: String| -> String {
        format!("{}{}", greeting, s)
    });

    // Closures

    call_a_thunk(|| println!("Hello world!"));

    call_this(|s| println!("{}", s));

    // External functions

    fn foo(s: &str) {}
    call_bare(foo);

    call_bare_again(foo);
}
