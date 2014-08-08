// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
  Functions for computing canonical and compatible decompositions
  for Unicode characters.
  */

use core::cmp::{Equal, Less, Greater};
use core::option::{Option, Some, None};
use core::slice::ImmutableVector;
use tables::normalization::{canonical_table, compatibility_table, composition_table};

fn bsearch_table<T>(c: char, r: &'static [(char, &'static [T])]) -> Option<&'static [T]> {
    match r.bsearch(|&(val, _)| {
        if c == val { Equal }
        else if val < c { Less }
        else { Greater }
    }) {
        Some(idx) => {
            let (_, result) = r[idx];
            Some(result)
        }
        None => None
    }
}

/// Compute canonical Unicode decomposition for character
pub fn decompose_canonical(c: char, i: |char|) { d(c, i, false); }

/// Compute canonical or compatible Unicode decomposition for character
pub fn decompose_compatible(c: char, i: |char|) { d(c, i, true); }

fn d(c: char, i: |char|, k: bool) {
    // 7-bit ASCII never decomposes
    if c <= '\x7f' { i(c); return; }

    // Perform decomposition for Hangul
    if (c as u32) >= S_BASE && (c as u32) < (S_BASE + S_COUNT) {
        decompose_hangul(c, i);
        return;
    }

    // First check the canonical decompositions
    match bsearch_table(c, canonical_table) {
        Some(canon) => {
            for x in canon.iter() {
                d(*x, |b| i(b), k);
            }
            return;
        }
        None => ()
    }

    // Bottom out if we're not doing compat.
    if !k { i(c); return; }

    // Then check the compatibility decompositions
    match bsearch_table(c, compatibility_table) {
        Some(compat) => {
            for x in compat.iter() {
                d(*x, |b| i(b), k);
            }
            return;
        }
        None => ()
    }

    // Finally bottom out.
    i(c);
}

pub fn compose(a: char, b: char) -> Option<char> {
    compose_hangul(a, b).or_else(|| {
        match bsearch_table(a, composition_table) {
            None => None,
            Some(candidates) => {
                match candidates.bsearch(|&(val, _)| {
                    if b == val { Equal }
                    else if val < b { Less }
                    else { Greater }
                }) {
                    Some(idx) => {
                        let (_, result) = candidates[idx];
                        Some(result)
                    }
                    None => None
                }
            }
        }
    })
}

// Constants from Unicode 6.3.0 Section 3.12 Conjoining Jamo Behavior
static S_BASE: u32 = 0xAC00;
static L_BASE: u32 = 0x1100;
static V_BASE: u32 = 0x1161;
static T_BASE: u32 = 0x11A7;
static L_COUNT: u32 = 19;
static V_COUNT: u32 = 21;
static T_COUNT: u32 = 28;
static N_COUNT: u32 = (V_COUNT * T_COUNT);
static S_COUNT: u32 = (L_COUNT * N_COUNT);

// Decompose a precomposed Hangul syllable
#[inline(always)]
fn decompose_hangul(s: char, f: |char|) {
    use core::mem::transmute;

    let si = s as u32 - S_BASE;

    let li = si / N_COUNT;
    unsafe {
        f(transmute(L_BASE + li));

        let vi = (si % N_COUNT) / T_COUNT;
        f(transmute(V_BASE + vi));

        let ti = si % T_COUNT;
        if ti > 0 {
            f(transmute(T_BASE + ti));
        }
    }
}

// Compose a pair of Hangul Jamo
#[inline(always)]
fn compose_hangul(a: char, b: char) -> Option<char> {
    use core::mem::transmute;
    let l = a as u32;
    let v = b as u32;
    // Compose an LPart and a VPart
    if L_BASE <= l && l < (L_BASE + L_COUNT) && V_BASE <= v && v < (V_BASE + V_COUNT) {
        let r = S_BASE + (l - L_BASE) * N_COUNT + (v - V_BASE) * T_COUNT;
        return unsafe { Some(transmute(r)) };
    }
    // Compose an LVPart and a TPart
    if S_BASE <= l && l <= (S_BASE+S_COUNT-T_COUNT) && T_BASE <= v && v < (T_BASE+T_COUNT) {
        let r = l + (v - T_BASE);
        return unsafe { Some(transmute(r)) };
    }
    None
}
