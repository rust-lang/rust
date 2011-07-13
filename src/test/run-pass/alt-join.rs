
use std;
import std::option;
import std::option::t;
import std::option::none;
import std::option::some;

fn foo[T](&option::t[T] y) {
    let int x;
    let vec[int] rs = [];
    /* tests that x doesn't get put in the precondition for the
       entire if expression */

    if (true) {
    } else {
        alt (y) { case (none[T]) { x = 17; } case (_) { x = 42; } }
        rs += [x];
    }
    ret;
}

fn main() { log "hello"; foo[int](some[int](5)); }