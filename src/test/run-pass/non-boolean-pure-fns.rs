use std;

import std::list::*;

pure fn pure_length_go<T>(ls: list<T>, acc: uint) -> uint {
    alt ls { nil. { acc } cons(_, tl) { pure_length_go(*tl, acc + 1u) } }
}

pure fn pure_length<T>(ls: list<T>) -> uint { pure_length_go(ls, 0u) }

pure fn nonempty_list<T>(ls: list<T>) -> bool { pure_length(ls) > 0u }

// Of course, the compiler can't take advantage of the
// knowledge that ls is a cons node. Future work.
// Also, this is pretty contrived since nonempty_list
// could be a "tag refinement", if we implement those.
fn safe_head<T>(ls: list<T>) : nonempty_list(ls) -> T { car(ls) }

fn main() {
    let mylist = cons(@1u, @nil);
    // Again, a way to eliminate such "obvious" checks seems
    // desirable. (Tags could have postconditions.)
    check (nonempty_list(mylist));
    assert (*safe_head(mylist) == 1u);
}
