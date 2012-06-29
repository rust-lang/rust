#[doc = "A standard linked list"];

import core::option;
import option::*;
import option::{some, none};

enum list<T> {
    cons(T, @list<T>),
    nil,
}

#[doc = "Create a list from a vector"]
fn from_vec<T: copy>(v: ~[T]) -> @list<T> {
    vec::foldr(v, @nil::<T>, { |h, t| @cons(h, t) })
}

#[doc = "
Left fold

Applies `f` to `u` and the first element in the list, then applies `f` to the
result of the previous call and the second element, and so on, returning the
accumulated result.

# Arguments

* ls - The list to fold
* z - The initial value
* f - The function to apply
"]
fn foldl<T: copy, U>(z: T, ls: @list<U>, f: fn(T, U) -> T) -> T {
    let mut accum: T = z;
    iter(ls) {|elt| accum = f(accum, elt);}
    accum
}

#[doc = "
Search for an element that matches a given predicate

Apply function `f` to each element of `v`, starting from the first.
When function `f` returns true then an option containing the element
is returned. If `f` matches no elements then none is returned.
"]
fn find<T: copy>(ls: @list<T>, f: fn(T) -> bool) -> option<T> {
    let mut ls = ls;
    loop {
        ls = alt *ls {
          cons(hd, tl) {
            if f(hd) { ret some(hd); }
            tl
          }
          nil { ret none; }
        }
    };
}

#[doc = "Returns true if a list contains an element with the given value"]
fn has<T: copy>(ls: @list<T>, elt: T) -> bool {
    for each(ls) { |e|
        if e == elt { ret true; }
    }
    ret false;
}

#[doc = "Returns true if the list is empty"]
pure fn is_empty<T: copy>(ls: @list<T>) -> bool {
    alt *ls {
        nil { true }
        _ { false }
    }
}

#[doc = "Returns true if the list is not empty"]
pure fn is_not_empty<T: copy>(ls: @list<T>) -> bool {
    ret !is_empty(ls);
}

#[doc = "Returns the length of a list"]
fn len<T>(ls: @list<T>) -> uint {
    let mut count = 0u;
    iter(ls) {|_e| count += 1u;}
    count
}

#[doc = "Returns all but the first element of a list"]
pure fn tail<T: copy>(ls: @list<T>) -> @list<T> {
    alt *ls {
        cons(_, tl) { ret tl; }
        nil { fail "list empty" }
    }
}

#[doc = "Returns the first element of a list"]
pure fn head<T: copy>(ls: @list<T>) -> T {
    alt check *ls { cons(hd, _) { hd } }
}

#[doc = "Appends one list to another"]
pure fn append<T: copy>(l: @list<T>, m: @list<T>) -> @list<T> {
    alt *l {
      nil { ret m; }
      cons(x, xs) { let rest = append(xs, m); ret @cons(x, rest); }
    }
}

#[doc = "Iterate over a list"]
fn iter<T>(l: @list<T>, f: fn(T)) {
    let mut cur = l;
    loop {
        cur = alt *cur {
          cons(hd, tl) {
            f(hd);
            tl
          }
          nil { break; }
        }
    }
}

#[doc = "Iterate over a list"]
fn each<T>(l: @list<T>, f: fn(T) -> bool) {
    let mut cur = l;
    loop {
        cur = alt *cur {
          cons(hd, tl) {
            if !f(hd) { ret; }
            tl
          }
          nil { break; }
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_is_empty() {
        let empty : @list::list<int> = from_vec(~[]);
        let full1 = from_vec(~[1]);
        let full2 = from_vec(~['r', 'u']);

        assert is_empty(empty);
        assert !is_empty(full1);
        assert !is_empty(full2);

        assert !is_not_empty(empty);
        assert is_not_empty(full1);
        assert is_not_empty(full2);
    }

    #[test]
    fn test_from_vec() {
        let l = from_vec(~[0, 1, 2]);

        assert (head(l) == 0);

        let tail_l = tail(l);
        assert (head(tail_l) == 1);

        let tail_tail_l = tail(tail_l);
        assert (head(tail_tail_l) == 2);
    }

    #[test]
    fn test_from_vec_empty() {
        let empty : @list::list<int> = from_vec(~[]);
        assert (empty == @list::nil::<int>);
    }

    #[test]
    fn test_foldl() {
        fn add(&&a: uint, &&b: int) -> uint { ret a + (b as uint); }
        let l = from_vec(~[0, 1, 2, 3, 4]);
        let empty = @list::nil::<int>;
        assert (list::foldl(0u, l, add) == 10u);
        assert (list::foldl(0u, empty, add) == 0u);
    }

    #[test]
    fn test_foldl2() {
        fn sub(&&a: int, &&b: int) -> int {
            a - b
        }
        let l = from_vec(~[1, 2, 3, 4]);
        assert (list::foldl(0, l, sub) == -10);
    }

    #[test]
    fn test_find_success() {
        fn match(&&i: int) -> bool { ret i == 2; }
        let l = from_vec(~[0, 1, 2]);
        assert (list::find(l, match) == option::some(2));
    }

    #[test]
    fn test_find_fail() {
        fn match(&&_i: int) -> bool { ret false; }
        let l = from_vec(~[0, 1, 2]);
        let empty = @list::nil::<int>;
        assert (list::find(l, match) == option::none::<int>);
        assert (list::find(empty, match) == option::none::<int>);
    }

    #[test]
    fn test_has() {
        let l = from_vec(~[5, 8, 6]);
        let empty = @list::nil::<int>;
        assert (list::has(l, 5));
        assert (!list::has(l, 7));
        assert (list::has(l, 8));
        assert (!list::has(empty, 5));
    }

    #[test]
    fn test_len() {
        let l = from_vec(~[0, 1, 2]);
        let empty = @list::nil::<int>;
        assert (list::len(l) == 3u);
        assert (list::len(empty) == 0u);
    }

}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
