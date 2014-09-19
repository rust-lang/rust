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
// from matches.

// All opening remarks from drop-obligations-1.rs also apply here.
// in particular:
//
// Note that this is not a true compile-fail test, in the sense that
// the code below is not meant to have any actual errors that are
// being detected by the comipiler. We are just re-purposing the
// compiletest error parsing as an easy way to annotate a test file
// with the expected operations from guts of the compiler.

// In all of the tests below, the arguments to the test are
// functions that:
//
// - d: move away an input
// - c: create an instance (usually of `T`)

pub enum E<X,Y> { Zero, One(X), Two(X,Y), }

#[rustc_drop_obligations]
fn matching_enum_by_move<T>(c: fn() -> E<T,T>, d: fn(T)) {
    // `matching_enum_by_move` shows how variant matching manipulates
    // the drop obligation state.

    {
        let e : E<T,T>;        //~  ERROR        move removes drop-obl `$(local e)`

        e = c();               //~  ERROR     assignment adds drop-obl `$(local e)`

        // (match arms do not have their own spans, which partly
        // explains why all the bindings for each arm are reported as
        // being scoped by the match itself.)

        match e {
            Zero => {          //~  ERROR    match whitelists drop-obl `$(local e)`

            }

            One(s_0) => {      //~  ERROR  refinement removes drop-obl `$(local e)`
                               //~^ ERROR         match adds drop-obl `($(local e)->One).#0u`
                               //~^^ ERROR      move removes drop-obl `($(local e)->One).#0u`
                               //~^^^ ERROR   assignment adds drop-obl `$(local s_0)`

                d(s_0);        //~  ERROR        move removes drop-obl `$(local s_0)`

            }

            Two(t_0, t_1) => { //~  ERROR   refinement removes drop-obl `$(local e)`
                               //~^ ERROR          match adds drop-obl `($(local e)->Two).#0u`
                               //~^^ ERROR       move removes drop-obl `($(local e)->Two).#0u`
                               //~^^^ ERROR    assignment adds drop-obl `$(local t_0)`
                               //~^^^^ ERROR       match adds drop-obl `($(local e)->Two).#1u`
                               //~^^^^^ ERROR    move removes drop-obl `($(local e)->Two).#1u`
                               //~^^^^^^ ERROR assignment adds drop-obl `$(local t_1)`

                d(t_0);        //~  ERROR         move removes drop-obl `$(local t_0)`
            }
        }                      //~   ERROR scope-end removes drop-obl `$(local s_0)`
                               //~^  ERROR scope-end removes drop-obl `$(local t_0)`
                               //~^^ ERROR scope-end removes drop-obl `$(local t_1)`


    }                          //~  ERROR       scope-end removes drop-obl `$(local e)`
                               //~^ ERROR      scope-end removes drop-obl `($(local e)->Zero)`
                               //~^^ ERROR     scope-end removes drop-obl `($(local e)->One).#0u`
                               //~^^^ ERROR    scope-end removes drop-obl `($(local e)->Two).#1u`
                               //~^^^^ ERROR   scope-end removes drop-obl `($(local e)->Two).#0u`
}

fn main() { }
