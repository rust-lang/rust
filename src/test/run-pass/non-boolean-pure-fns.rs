use std;

use std::list::*;

pure fn pure_length_go<T: Copy>(ls: @List<T>, acc: uint) -> uint {
    match *ls { Nil => { acc } Cons(_, tl) => { pure_length_go(tl, acc + 1u) } }
}

pure fn pure_length<T: Copy>(ls: @List<T>) -> uint { pure_length_go(ls, 0u) }

pure fn nonempty_list<T: Copy>(ls: @List<T>) -> bool { pure_length(ls) > 0u }

fn safe_head<T: Copy>(ls: @List<T>) -> T {
    assert is_not_empty(ls);
    return head(ls);
}

fn main() {
    let mylist = @Cons(@1u, @Nil);
    assert (nonempty_list(mylist));
    assert (*safe_head(mylist) == 1u);
}
