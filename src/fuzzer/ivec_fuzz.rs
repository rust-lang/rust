/*

Idea: provide functions for 'exhaustive' and 'random' modification of vecs.

  two functions, "return all edits" and "return a random edit" <-- leaning toward this model
    or
  two functions, "return the number of possible edits" and "return edit #n"

It would be nice if this could be data-driven, so the two functions could share information:
  type vec_modifier = rec(fn (&<T> v, uint i) -> [T] fun, uint lo, uint di);
  const [vec_modifier] vec_modifiers = ~[rec(fun=vec_omit, 0u, 1u), ...];
But that gives me "error: internal compiler error unimplemented consts that's not a plain literal".
https://github.com/graydon/rust/issues/570

vec_edits is not an iter because iters might go away.

*/

use std;
import std::vec;
import std::vec::slice;
import std::vec::len;
import std::int;

//fn vec_reverse(&<T> v) -> [T] { ... }

fn vec_omit<@T>(v: &[T], i: uint) -> [T] {
    slice(v, 0u, i) + slice(v, i + 1u, len(v))
}
fn vec_dup<@T>(v: &[T], i: uint) -> [T] {
    slice(v, 0u, i) + [v[i]] + slice(v, i, len(v))
}
fn vec_swadj<@T>(v: &[T], i: uint) -> [T] {
    slice(v, 0u, i) + [v[i + 1u], v[i]] + slice(v, i + 2u, len(v))
}
fn vec_prefix<@T>(v: &[T], i: uint) -> [T] { slice(v, 0u, i) }
fn vec_suffix<@T>(v: &[T], i: uint) -> [T] { slice(v, i, len(v)) }

fn vec_poke<@T>(v: &[T], i: uint, x: &T) -> [T] {
    slice(v, 0u, i) + [x] + slice(v, i + 1u, len(v))
}
fn vec_insert<@T>(v: &[T], i: uint, x: &T) -> [T] {
    slice(v, 0u, i) + [x] + slice(v, i, len(v))
}

// Iterates over 0...length, skipping the specified number on each side.
iter ix(skip_low: uint, skip_high: uint, length: uint) -> uint {
    let i: uint = skip_low;
    while i + skip_high <= length { put i; i += 1u; }
}

// Returns a bunch of modified versions of v, some of which introduce new elements (borrowed from xs).
fn vec_edits<@T>(v: &[T], xs: &[T]) -> [[T]] {
    let edits: [[T]] = [];
    let Lv: uint = len(v);

    if Lv != 1u {
        edits +=
            [[]]; // When Lv == 1u, this is redundant with omit
                  //if (Lv >= 3u) { edits += ~[vec_reverse(v)]; }



    }
    for each i: uint in ix(0u, 1u, Lv) { edits += [vec_omit(v, i)]; }
    for each i: uint in ix(0u, 1u, Lv) { edits += [vec_dup(v, i)]; }
    for each i: uint in ix(0u, 2u, Lv) { edits += [vec_swadj(v, i)]; }
    for each i: uint in ix(1u, 2u, Lv) { edits += [vec_prefix(v, i)]; }
    for each i: uint in ix(2u, 1u, Lv) { edits += [vec_suffix(v, i)]; }

    for each j: uint in ix(0u, 1u, len(xs)) {
        for each i: uint in ix(0u, 1u, Lv) {
            edits += [vec_poke(v, i, xs[j])];
        }
        for each i: uint in ix(0u, 0u, Lv) {
            edits += [vec_insert(v, i, xs[j])];
        }
    }


    edits
}

// Would be nice if this were built in: https://github.com/graydon/rust/issues/424
fn vec_to_str(v: &[int]) -> str {
    let i = 0u;
    let s = "[";
    while i < len(v) {
        s += int::str(v[i]);
        if i + 1u < len(v) { s += ", " }
        i += 1u;
    }
    ret s + "]";
}

fn show_edits(a: &[int], xs: &[int]) {
    log_err "=== Edits of " + vec_to_str(a) + " ===";
    let b = vec_edits(a, xs);
    for each i: uint in ix(0u, 1u, len(b)) { log_err vec_to_str(b[i]); }
}

fn demo_edits() {
    let xs = [7, 8];
    show_edits([], xs);
    show_edits([1], xs);
    show_edits([1, 2], xs);
    show_edits([1, 2, 3], xs);
    show_edits([1, 2, 3, 4], xs);
}

fn main() { demo_edits(); }
