/*

Idea: provide functions for 'exhaustive' and 'random' modification of vecs.

  two functions, "return all edits" and "return a random edit" <-- leaning toward this model
    or
  two functions, "return the number of possible edits" and "return edit #n"

It would be nice if this could be data-driven, so the two functions could share information:
  type vec_modifier = rec(fn (&int[] v, uint i) -> int[] fun, uint lo, uint di);
  const vec_modifier[] vec_modifiers = ~[rec(fun=vec_omit, 0u, 1u), ...];
But that gives me "error: internal compiler error unimplemented consts that's not a plain literal".
https://github.com/graydon/rust/issues/570

vec_edits is not an iter because iters might go away and:
https://github.com/graydon/rust/issues/639

vec_omit and friends are not type-parameterized because:
https://github.com/graydon/rust/issues/640

*/

use std;
import std::ivec;
import std::ivec::slice;
import std::ivec::len;
import std::int;

//fn vec_reverse(&int[] v) -> int[] { ... }

fn vec_omit   (&int[] v, uint i) -> int[] { slice(v, 0u, i) +                      slice(v, i+1u, len(v)) }
fn vec_dup    (&int[] v, uint i) -> int[] { slice(v, 0u, i) + ~[v.(i)]           + slice(v, i,    len(v)) }
fn vec_swadj  (&int[] v, uint i) -> int[] { slice(v, 0u, i) + ~[v.(i+1u), v.(i)] + slice(v, i+2u, len(v)) }
fn vec_prefix (&int[] v, uint i) -> int[] { slice(v, 0u, i) }
fn vec_suffix (&int[] v, uint i) -> int[] { slice(v, i, len(v)) }

fn vec_poke   (&int[] v, uint i, int x) -> int[] { slice(v, 0u, i) + ~[x] + slice(v, i+1u, len(v)) }
fn vec_insert (&int[] v, uint i, int x) -> int[] { slice(v, 0u, i) + ~[x] + slice(v, i, len(v)) }

// Iterates over 0...length, skipping the specified number on each side.
iter ix(uint skip_low, uint skip_high, uint length) -> uint { let uint i = skip_low; while (i + skip_high <= length) { put i; i += 1u; } }

// Returns a bunch of modified versions of v, some of which introduce new elements (borrowed from xs).
fn vec_edits(&int[] v, &int[] xs) -> int[][] {
    let int[][] edits = ~[];
    let uint Lv = len(v);

    if (Lv != 1u) { edits += ~[~[]]; } // When Lv == 1u, this is redundant with omit
    //if (Lv >= 3u) { edits += ~[vec_reverse(v)]; }

    for each (uint i in ix(0u, 1u, Lv)) { edits += ~[vec_omit  (v, i)]; }
    for each (uint i in ix(0u, 1u, Lv)) { edits += ~[vec_dup   (v, i)]; }
    for each (uint i in ix(0u, 2u, Lv)) { edits += ~[vec_swadj (v, i)]; }
    for each (uint i in ix(1u, 2u, Lv)) { edits += ~[vec_prefix(v, i)]; }
    for each (uint i in ix(2u, 1u, Lv)) { edits += ~[vec_suffix(v, i)]; }

    for each (uint j in ix(0u, 1u, len(xs))) {
      for each (uint i in ix(0u, 1u, Lv)) { edits += ~[vec_poke  (v, i, xs.(j))]; }
      for each (uint i in ix(0u, 0u, Lv)) { edits += ~[vec_insert(v, i, xs.(j))]; }
    }

    edits
}

// Would be nice if this were built in: https://github.com/graydon/rust/issues/424
fn vec_to_str(&int[] v) -> str {
    auto i = 0u;
    auto s = "[";
    while (i < len(v)) {
        s += int::str(v.(i));
        if (i + 1u < len(v)) {
            s += ", "
        }
        i += 1u;
    }
    ret s + "]";
}

fn show_edits(&int[] a, &int[] xs) {
    log_err "=== Edits of " + vec_to_str(a) + " ===";
    auto b = vec_edits(a, xs);
    for each (uint i in ix(0u, 1u, len(b))) {
        log_err vec_to_str(b.(i));
    }
}

fn demo_edits() {
    auto xs = ~[7, 8];
    show_edits(~[], xs);
    show_edits(~[1], xs);
    show_edits(~[1,2], xs);
    show_edits(~[1,2,3], xs);
    show_edits(~[1,2,3,4], xs);
}

fn main() {
    demo_edits();
}
