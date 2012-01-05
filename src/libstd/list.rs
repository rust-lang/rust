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
fn from_vec<T: copy>(v: [const T]) -> list<T> {
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
fn foldl<T: copy, U>(ls: list<U>, z: T, f: block(T, U) -> T) -> T {
    let accum: T = z;
    iter(ls) {|elt| accum = f(accum, elt);}
    accum
}

/*
Function: find

Search for an element that matches a given predicate

Apply function `f` to each element of `v`, starting from the first.
When function `f` returns true then an option containing the element
is returned. If `f` matches no elements then none is returned.
*/
fn find<T: copy, U: copy>(ls: list<T>, f: block(T) -> option::t<U>)
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
fn has<T: copy>(ls: list<T>, elt: T) -> bool {
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
Function: is_empty

Returns true if the list is empty.
*/
pure fn is_empty<T: copy>(ls: list<T>) -> bool {
    alt ls {
        nil. { true }
        _ { false }
    }
}

/*
Function: is_not_empty

Returns true if the list is not empty.
*/
pure fn is_not_empty<T: copy>(ls: list<T>) -> bool {
    ret !is_empty(ls);
}

/*
Function: len

Returns the length of a list
*/
fn len<T>(ls: list<T>) -> uint {
    let count = 0u;
    iter(ls) {|_e| count += 1u;}
    count
}

/*
Function: tail

Returns all but the first element of a list
*/
pure fn tail<T: copy>(ls: list<T>) : is_not_empty(ls) -> list<T> {
    alt ls {
        cons(_, tl) { ret *tl; }
        nil. { fail "list empty" }
    }
}

/*
Function: head

Returns the first element of a list
*/
pure fn head<T: copy>(ls: list<T>) : is_not_empty(ls) -> T {
    alt ls {
        cons(hd, _) { ret hd; }
        nil. { fail "list empty" }
    }
}

/*
Function: append

Appends one list to another
*/
pure fn append<T: copy>(l: list<T>, m: list<T>) -> list<T> {
    alt l {
      nil. { ret m; }
      cons(x, xs) { let rest = append(*xs, m); ret cons(x, @rest); }
    }
}

/*
Function: iter

Iterate over a list
*/
fn iter<T>(l: list<T>, f: block(T)) {
    alt l {
      cons(hd, tl) {
        f(hd);
        let cur = tl;
        while true {
            alt *cur {
              cons(hd, tl) {
                f(hd);
                cur = tl;
              }
              nil. { break; }
            }
        }
      }
      nil. {}
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
