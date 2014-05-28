// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

use std::mem;
use std::io::stdio::println;

fn call_it(f: proc(String) -> String) {
    println!("{}", f("Fred".to_string()))
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

    let greeting = "Hello ".to_string();
    call_it(proc(s) {
        format_strbuf!("{}{}", greeting, s)
    });

    let greeting = "Goodbye ".to_string();
    call_it(proc(s) format_strbuf!("{}{}", greeting, s));

    let greeting = "How's life, ".to_string();
    call_it(proc(s: String) -> String {
        format_strbuf!("{}{}", greeting, s)
    });

    // Closures

    call_a_thunk(|| println!("Hello world!"));

    call_this(|s| println!("{}", s));

    call_that(|x, y| *x + *y);

    let z = 100;
    call_that(|x, y| *x + *y - z);

    call_cramped(|| 1, || unsafe {
        static a: uint = 100;
        mem::transmute(&a)
    });

    // External functions

    call_bare(println);

    call_bare_again(println);
}

