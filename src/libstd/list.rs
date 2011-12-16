/*
Module: list

A standard linked list
*/

import core::option;
import option::*;
import option::{some, none};

/* Section: Types */

/*
Tag: list
*/
tag list<T> {
    /* Variant: cons */
    cons(T, @list<T>);
    /* Variant: nil */
    nil;
}

/*Section: Operations */

/*
Function: from_vec

Create a list from a vector
*/
fn from_vec<copy T>(v: [const T]) -> list<T> {
    *vec::foldr(v, @nil::<T>, { |h, t| @cons(h, t) })
}

/*
Function: foldl

Left fold

Applies `f` to `u` and the first element in the list, then applies
`f` to the result of the previous call and the second element,
and so on, returning the accumulated result.

Parameters:

ls - The list to fold
z - The initial value
f - The function to apply
*/
fn foldl<copy T, copy U>(ls: list<U>, z: T, f: block(T, U) -> T) -> T {
    let accum: T = z;
    let ls = ls;
    while true {
        alt ls {
          cons(hd, tl) { accum = f(accum, hd); ls = *tl; }
          nil. { break; }
        }
    }
    ret accum;
}

/*
Function: find

Search for an element that matches a given predicate

Apply function `f` to each element of `v`, starting from the first.
When function `f` returns true then an option containing the element
is returned. If `f` matches no elements then none is returned.
*/
fn find<copy T, copy U>(ls: list<T>, f: block(T) -> option::t<U>)
    -> option::t<U> {
    let ls = ls;
    while true {
        alt ls {
          cons(hd, tl) {
            alt f(hd) { none. { ls = *tl; } some(rs) { ret some(rs); } }
          }
          nil. { break; }
        }
    }
    ret none;
}

/*
Function: has

Returns true if a list contains an element with the given value
*/
fn has<copy T>(ls: list<T>, elt: T) -> bool {
    let ls = ls;
    while true {
        alt ls {
          cons(hd, tl) { if elt == hd { ret true; } else { ls = *tl; } }
          nil. { break; }
        }
    }
    ret false;
}

/*
Function: len

Returns the length of a list
*/
fn len<copy T>(ls: list<T>) -> uint {
    fn count<T>(&&u: uint, _t: T) -> uint { ret u + 1u; }
    ret foldl(ls, 0u, bind count(_, _));
}

/*
Function: tail

Returns all but the first element of a list
*/
pure fn tail<copy T>(ls: list<T>) -> list<T> {
    alt ls { cons(_, tl) { ret *tl; } nil. { fail "list empty" } }
}

/*
Function: head

Returns the first element of a list
*/
pure fn head<copy T>(ls: list<T>) -> T {
    alt ls { cons(hd, _) { ret hd; } nil. { fail "list empty" } }
}

/*
Function: append

Appends one list to another
*/
pure fn append<copy T>(l: list<T>, m: list<T>) -> list<T> {
    alt l {
      nil. { ret m; }
      cons(x, xs) { let rest = append(*xs, m); ret cons(x, @rest); }
    }
}

/*
Function: iter

Iterate over a list
*/
fn iter<copy T>(l: list<T>, f: block(T)) {
    let cur = l;
    while cur != nil {
        alt cur {
          cons(hd, tl) {
            f(hd);
            cur = *tl;
          }
        }
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
