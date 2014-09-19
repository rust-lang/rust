// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we correctly compute the changes to drop obligations
// from bindings, moves, and assignments.

// Note that this is not a true compile-fail test, in the sense that
// the code below is not meant to have any actual errors that are
// being detected by the comipiler. We are just re-purposing the
// compiletest error parsing as an easy way to annotate a test file
// with the expected operations from guts of the compiler.
//
// Note also that this is *not* testing the control-flow analysis
// itself (i.e. the computation of the actual drop-obligations at each
// control-flow merge point).  It is just testing (and, perhaps more
// importantly, showing rustc developers) how each expression will
// affect the eventual control-flow analysis, i.e. what are the
// so-called "gen/kill" sets, which are inputs that drive the
// control-flow analysis.
//
// This latter point means that it is not so important here to check
// how things like if-expressions and loops work, since those tend to
// only matter in the control-flow analysis itself. But binding,
// assignment, and referencing constructs like `let`, `match`, and
// field-deref all *do* matter in this test.

// The tests are written in terms of a generic type T for the values
// being moved/dropped since that is the simplest way to represent a
// potential drop obligation.

// In all of the tests below, the arguments to the test are
// functions that:
//
// - d: move away an input
// - c: create an instance (usually of `T`)

#[rustc_drop_obligations]
fn simplest<T>(d: fn(T), c: fn() -> T)
{
    {
        let x: T;   //~  ERROR      move removes drop-obl `$(local x)`
        let y: T;   //~  ERROR      move removes drop-obl `$(local y)`
        x = c();    //~  ERROR   assignment adds drop-obl `$(local x)`

    }               //~  ERROR scope-end removes drop-obl `$(local x)`
                    //~^ ERROR scope-end removes drop-obl `$(local y)`

}

struct Pair<X,Y> { x: X, y: Y }

#[rustc_drop_obligations]
fn struct_as_unit<T>(d: fn(Pair<T,T>), c: fn() -> T)
{
    // `struct_as_unit` is illustrating how a struct like `Pair<T,T>`
    // is treated as an atomic entity if you do not move its fields
    // independently from the struct as a whole.

    {
        let p;      //~  ERROR      move removes drop-obl `$(local p)`

        p = Pair {  //~  ERROR   assignment adds drop-obl `$(local p)`
            x: c(), y: c()
        };

        d(p);       //~  ERROR      move removes drop-obl `$(local p)`

    }               //~  ERROR scope-end removes drop-obl `$(local p)`

}

#[rustc_drop_obligations]
fn struct_decomposes<T>(d: fn(T), c: fn() -> T)
{
    // `struct_decomposes` is illustrating how `Pair<T,T>` will break
    // down into fragment for each individual drop obligation if you
    // do move its fields independently from the struct as a whole.

    {
        let p;      //~  ERROR      move removes drop-obl `$(local p).x`
                    //~^ ERROR      move removes drop-obl `$(local p).y`

        p = Pair {  //~  ERROR   assignment adds drop-obl `$(local p).x`
                    //~^ ERROR   assignment adds drop-obl `$(local p).y`
            x: c(), y: c()
        };

        d(p.x);     //~  ERROR      move removes drop-obl `$(local p).x`

    }               //~  ERROR scope-end removes drop-obl `$(local p).x`
                    //~^ ERROR scope-end removes drop-obl `$(local p).y`
}

#[rustc_drop_obligations]
fn struct_copy_fields_ignorable<T,K:Copy>(d_t: fn(T), d_k: fn(K), c: fn() -> Pair<T,K>)
{
    // `struct_copy_fields_ignorable` illustrates copying a field
    // (i.e. moving one that implements Copy) does not cause the
    // struct to fragment.

    {
        let p;      //~  ERROR      move removes drop-obl `$(local p)`

        p = c();    //~  ERROR   assignment adds drop-obl `$(local p)`

        d_k(p.y);

    }               //~  ERROR scope-end removes drop-obl `$(local p)`
}

#[rustc_drop_obligations]
fn simple_enum<T>(c: fn() -> Option<T>, d: fn(Option<T>)) {
    // `simple_enum` just shows the basic (non-match) operations work
    // on enum types.
    //
    // Note in particular that, in the current analysis, since we do
    // not have refinement types, initializing `e` with a Copy variant
    // like `None` (or `Zero` from E above) still introduces a
    // drop-obligation on `e`.

    {
        let e : Option<T>;  //~  ERROR      move removes drop-obl `$(local e)`

        e = None;           //~  ERROR   assignment adds drop-obl `$(local e)`

        d(e);               //~  ERROR      move removes drop-obl `$(local e)`

    }                       //~  ERROR scope-end removes drop-obl `$(local e)`
}

fn main() { }
