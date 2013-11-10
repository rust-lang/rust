/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

use std::cast;

fn call_it(f: proc(~str) -> ~str) {
    println(f(~"Fred"))
}

fn call_a_thunk(f: ||) {
    f();
}

fn call_this(f: |&str|:Send) {
    f("Hello!");
}

fn call_that(f: <'a>|&'a int, &'a int|: -> int) {
    let (ten, forty_two) = (10, 42);
    println!("Your lucky number is {}", f(&ten, &forty_two));
}

fn call_cramped(f:||->uint,g:<'a>||->&'a uint) {
    let number = f();
    let other_number = *g();
    println!("Ticket {} wins an all-expenses-paid trip to Mountain View", number + other_number);
}

fn call_bare(f: fn(&str)) {
    f("Hello world!")
}

fn call_bare_again(f: extern "Rust" fn(&str)) {
    f("Goodbye world!")
}

pub fn main() {
    // Procs

    let greeting = ~"Hi ";
    do call_it |s| {
        greeting + s
    }

    let greeting = ~"Hello ";
    call_it(proc(s) {
        greeting + s
    });

    let greeting = ~"Goodbye ";
    call_it(proc(s) greeting + s);

    let greeting = ~"How's life, ";
    call_it(proc(s: ~str) -> ~str {
        greeting + s
    });

    // Closures

    call_a_thunk(|| println("Hello world!"));

    call_this(|s| println(s));

    call_that(|x, y| *x + *y);

    let z = 100;
    call_that(|x, y| *x + *y - z);

    call_cramped(|| 1, || unsafe {
        static a: uint = 100;
        cast::transmute(&a)
    });

    // External functions

    call_bare(println);

    call_bare_again(println);
}

