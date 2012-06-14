use std;

import std::list::*;

pure fn pure_length_go<T: copy>(ls: @list<T>, acc: uint) -> uint {
    alt *ls { nil { acc } cons(_, tl) { pure_length_go(tl, acc + 1u) } }
}

pure fn pure_length<T: copy>(ls: @list<T>) -> uint { pure_length_go(ls, 0u) }

pure fn nonempty_list<T: copy>(ls: @list<T>) -> bool { pure_length(ls) > 0u }

// Of course, the compiler can't take advantage of the
// knowledge that ls is a cons node. Future work.
// Also, this is pretty contrived since nonempty_list
// could be a "enum refinement", if we implement those.
fn safe_head<T: copy>(ls: @list<T>) -> T {
    assert is_not_empty(ls);
    ret head(ls);
}

fn main() {
    let mylist = @cons(@1u, @nil);
    assert (nonempty_list(mylist));
    assert (*safe_head(mylist) == 1u);
}
