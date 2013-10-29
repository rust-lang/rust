/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

fn call_it(f: proc(~str) -> ~str) {
    println(f(~"Fred"))
}

pub fn main() {
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
}

