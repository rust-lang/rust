//@ revisions: stable feature
#![cfg_attr(feature, feature(structural_init))]

// This test enumerates various cases of interest where an ADT or tuple is
// partially initialized and then used in some way that is wrong *even*
// with `feature(structural_init)`. The error output in both cases should be identical.
//
// See structural_init.rs for cases of tests that are
// meant to compile and run successfully with `structural_init`.

struct D {
    x: u32,
    s: S,
}

struct S {
    y: u32,
    z: u32,
}


impl Drop for D {
    fn drop(&mut self) { }
}

fn cannot_partially_init_adt_with_drop() {
    let d: D;
    d.x = 10; //~ ERROR E0381
}

fn cannot_partially_init_mutable_adt_with_drop() {
    let mut d: D;
    d.x = 10; //~ ERROR E0381
}

fn cannot_partially_reinit_adt_with_drop() {
    let mut d = D { x: 0, s: S{ y: 0, z: 0 } };
    drop(d);
    d.x = 10;
    //~^ ERROR assign of moved value: `d` [E0382]
}

fn cannot_partially_init_inner_adt_via_outer_with_drop() {
    let d: D;
    d.s.y = 20; //~ ERROR E0381
}

fn cannot_partially_init_inner_adt_via_mutable_outer_with_drop() {
    let mut d: D;
    d.s.y = 20; //~ ERROR E0381
}

fn cannot_partially_reinit_inner_adt_via_outer_with_drop() {
    let mut d = D { x: 0, s: S{ y: 0, z: 0} };
    drop(d);
    d.s.y = 20;
    //[stable]~^ ERROR assign to part of moved value: `d` [E0382]
    //[feature]~^^ ERROR assign of moved value: `d` [E0382]
    // FIXME: nonsense diagnostic
}

fn main() { }
