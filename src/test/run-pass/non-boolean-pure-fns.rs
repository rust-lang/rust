use std;

import std::list::*;

pure fn pure_length_go<T: copy>(ls: @list<T>, acc: uint) -> uint {
    match *ls { nil => { acc } cons(_, tl) => { pure_length_go(tl, acc + 1u) } }
}

pure fn pure_length<T: copy>(ls: @list<T>) -> uint { pure_length_go(ls, 0u) }

pure fn nonempty_list<T: copy>(ls: @list<T>) -> bool { pure_length(ls) > 0u }

fn safe_head<T: copy>(ls: @list<T>) -> T {
    assert is_not_empty(ls);
    return head(ls);
}

fn main() {
    let mylist = @cons(@1u, @nil);
    assert (nonempty_list(mylist));
    assert (*safe_head(mylist) == 1u);
}
