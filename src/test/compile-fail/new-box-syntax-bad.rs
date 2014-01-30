/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

// Tests that the new `box` syntax works with unique pointers and GC pointers.

use std::gc::Gc;
use std::owned::HEAP;

pub fn main() {
    let x: Gc<int> = box(HEAP) 2;  //~ ERROR mismatched types
    let y: Gc<int> = box(HEAP)(1 + 2);  //~ ERROR mismatched types
    let z: ~int = box(GC)(4 + 5);   //~ ERROR mismatched types
}

