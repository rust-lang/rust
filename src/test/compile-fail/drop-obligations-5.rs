// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we respect both `|\` and `proc` closures.

// All opening remarks from drop-obligations-1.rs also apply here.
// in particular:
//
// Note that this is not a true compile-fail test, in the sense that
// the code below is not meant to have any actual errors that are
// being detected by the comipiler. We are just re-purposing the
// compiletest error parsing as an easy way to annotate a test file
// with the expected operations from guts of the compiler.

// In all of the tests below, the three arguments to the test are
// functions that:
//
// - c: create an instance (usually of `T`)
// - i: does "something" with a given closure
// - b: does "something" wih borrowed `&T`
// - m: does "something" wih borrowed `&mut T`

#[rustc_drop_obligations]
fn stack_closure<T>(c: fn() -> T, i: fn(||), b: fn (&T))
{
    {
        let x: T;   //~  ERROR      move removes drop-obl `$(local x)`
        let y: T;   //~  ERROR      move removes drop-obl `$(local y)`
        x = c();    //~  ERROR   assignment adds drop-obl `$(local x)`

        // A || captures free-variables by reference, so we
        // get no note here.
        i(|| { b(&x); });

    }               //~  ERROR scope-end removes drop-obl `$(local x)`
                    //~^ ERROR scope-end removes drop-obl `$(local y)`

}

#[rustc_drop_obligations]
fn proc_closure_1<T>(c: fn() -> T, i: fn(proc ()))
{
    {
        let x: T;   //~  ERROR      move removes drop-obl `$(local x)`
        let y: T;   //~  ERROR      move removes drop-obl `$(local y)`
        x = c();    //~  ERROR   assignment adds drop-obl `$(local x)`

        // A proc captures free-variables by moving them, so we
        // get a note here.

        i(proc() { x; });
                    //~^ ERROR      move removes drop-obl `$(local x)`

    }               //~  ERROR scope-end removes drop-obl `$(local x)`
                    //~^ ERROR scope-end removes drop-obl `$(local y)`

}

#[rustc_drop_obligations]
fn proc_closure_2<T>(c: fn() -> T, i: fn(proc ()), b: fn (&T))
{
    {
        let x: T;   //~  ERROR      move removes drop-obl `$(local x)`
        let y: T;   //~  ERROR      move removes drop-obl `$(local y)`
        x = c();    //~  ERROR   assignment adds drop-obl `$(local x)`

        // A proc captures free-variables by moving them, so we
        // get a note here.

        i(proc() { b(&x); });
                    //~^ ERROR      move removes drop-obl `$(local x)`

    }               //~  ERROR scope-end removes drop-obl `$(local x)`
                    //~^ ERROR scope-end removes drop-obl `$(local y)`

}

fn main() { }
