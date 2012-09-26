//! A standard linked list
#[warn(deprecated_mode)];
#[forbid(deprecated_pattern)];

use core::cmp::Eq;
use core::option;
use option::*;
use option::{Some, None};

enum List<T> {
    Cons(T, @List<T>),
    Nil,
}

/// Cregate a list from a vector
fn from_vec<T: Copy>(v: &[T]) -> @List<T> {
    vec::foldr(v, @Nil::<T>, |h, t| @Cons(h, t))
}

/**
 * Left fold
 *
 * Applies `f` to `u` and the first element in the list, then applies `f` to
 * the result of the previous call and the second element, and so on,
 * returning the accumulated result.
 *
 * # Arguments
 *
 * * ls - The list to fold
 * * z - The initial value
 * * f - The function to apply
 */
fn foldl<T: Copy, U>(+z: T, ls: @List<U>, f: fn((&T), (&U)) -> T) -> T {
    let mut accum: T = z;
    do iter(ls) |elt| { accum = f(&accum, elt);}
    accum
}

/**
 * Search for an element that matches a given predicate
 *
 * Apply function `f` to each element of `v`, starting from the first.
 * When function `f` returns true then an option containing the element
 * is returned. If `f` matches no elements then none is returned.
 */
fn find<T: Copy>(ls: @List<T>, f: fn((&T)) -> bool) -> Option<T> {
    let mut ls = ls;
    loop {
        ls = match *ls {
          Cons(hd, tl) => {
            if f(&hd) { return Some(hd); }
            tl
          }
          Nil => return None
        }
    };
}

/// Returns true if a list contains an element with the given value
fn has<T: Copy Eq>(ls: @List<T>, +elt: T) -> bool {
    for each(ls) |e| {
        if e == elt { return true; }
    }
    return false;
}

/// Returns true if the list is empty
pure fn is_empty<T: Copy>(ls: @List<T>) -> bool {
    match *ls {
        Nil => true,
        _ => false
    }
}

/// Returns true if the list is not empty
pure fn is_not_empty<T: Copy>(ls: @List<T>) -> bool {
    return !is_empty(ls);
}

/// Returns the length of a list
fn len<T>(ls: @List<T>) -> uint {
    let mut count = 0u;
    iter(ls, |_e| count += 1u);
    count
}

/// Returns all but the first element of a list
pure fn tail<T: Copy>(ls: @List<T>) -> @List<T> {
    match *ls {
        Cons(_, tl) => return tl,
        Nil => fail ~"list empty"
    }
}

/// Returns the first element of a list
pure fn head<T: Copy>(ls: @List<T>) -> T {
    match *ls {
      Cons(hd, _) => hd,
      // makes me sad
      _ => fail ~"head invoked on empty list"
    }
}

/// Appends one list to another
pure fn append<T: Copy>(l: @List<T>, m: @List<T>) -> @List<T> {
    match *l {
      Nil => return m,
      Cons(x, xs) => {
        let rest = append(xs, m);
        return @Cons(x, rest);
      }
    }
}

/*
/// Push one element into the front of a list, returning a new list
/// THIS VERSION DOESN'T ACTUALLY WORK
pure fn push<T: Copy>(ll: &mut @list<T>, +vv: T) {
    ll = &mut @cons(vv, *ll)
}
*/

/// Iterate over a list
fn iter<T>(l: @List<T>, f: fn((&T))) {
    let mut cur = l;
    loop {
        cur = match *cur {
          Cons(ref hd, tl) => {
            f(hd);
            tl
          }
          Nil => break
        }
    }
}

/// Iterate over a list
fn each<T>(l: @List<T>, f: fn(T) -> bool) {
    let mut cur = l;
    loop {
        cur = match *cur {
          Cons(hd, tl) => {
            if !f(hd) { return; }
            tl
          }
          Nil => break
        }
    }
}

impl<T:Eq> List<T> : Eq {
    pure fn eq(other: &List<T>) -> bool {
        match self {
            Cons(e0a, e1a) => {
                match (*other) {
                    Cons(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            Nil => {
                match (*other) {
                    Nil => true,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &List<T>) -> bool { !self.eq(other) }
}

#[cfg(test)]
mod tests {
    #[legacy_exports];

    #[test]
    fn test_is_empty() {
        let empty : @list::List<int> = from_vec(~[]);
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
        let empty : @list::List<int> = from_vec(~[]);
        assert (empty == @list::Nil::<int>);
    }

    #[test]
    fn test_foldl() {
        fn add(a: &uint, b: &int) -> uint { return *a + (*b as uint); }
        let l = from_vec(~[0, 1, 2, 3, 4]);
        let empty = @list::Nil::<int>;
        assert (list::foldl(0u, l, add) == 10u);
        assert (list::foldl(0u, empty, add) == 0u);
    }

    #[test]
    fn test_foldl2() {
        fn sub(a: &int, b: &int) -> int {
            *a - *b
        }
        let l = from_vec(~[1, 2, 3, 4]);
        assert (list::foldl(0, l, sub) == -10);
    }

    #[test]
    fn test_find_success() {
        fn match_(i: &int) -> bool { return *i == 2; }
        let l = from_vec(~[0, 1, 2]);
        assert (list::find(l, match_) == option::Some(2));
    }

    #[test]
    fn test_find_fail() {
        fn match_(_i: &int) -> bool { return false; }
        let l = from_vec(~[0, 1, 2]);
        let empty = @list::Nil::<int>;
        assert (list::find(l, match_) == option::None::<int>);
        assert (list::find(empty, match_) == option::None::<int>);
    }

    #[test]
    fn test_has() {
        let l = from_vec(~[5, 8, 6]);
        let empty = @list::Nil::<int>;
        assert (list::has(l, 5));
        assert (!list::has(l, 7));
        assert (list::has(l, 8));
        assert (!list::has(empty, 5));
    }

    #[test]
    fn test_len() {
        let l = from_vec(~[0, 1, 2]);
        let empty = @list::Nil::<int>;
        assert (list::len(l) == 3u);
        assert (list::len(empty) == 0u);
    }

    #[test]
    fn test_append() {
        assert from_vec(~[1,2,3,4])
            == list::append(list::from_vec(~[1,2]), list::from_vec(~[3,4]));
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
