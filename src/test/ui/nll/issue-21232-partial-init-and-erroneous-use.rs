// This test enumerates various cases of interest where an ADT or tuple is
// partially initialized and then used in some way that is wrong *even*
// after rust-lang/rust#54987 is implemented.
//
// See rust-lang/rust#21232, rust-lang/rust#54986, and rust-lang/rust#54987.
//
// See issue-21232-partial-init-and-use.rs for cases of tests that are
// meant to compile and run successfully once rust-lang/rust#54987 is
// implemented.

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
    //~^ ERROR assign to part of moved value: `d` [E0382]
}

fn main() { }
