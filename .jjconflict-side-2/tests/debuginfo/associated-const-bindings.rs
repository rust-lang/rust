//@ compile-flags: -g
//@ disable-gdb-pretty-printers
//@ ignore-backends: gcc

//@ gdb-command:run
//@ gdb-command:whatis local
//@ gdb-check:type = &dyn associated_const_bindings::Trait<N=101>

//@ cdb-command: g
//@ cdb-command:dv /t /n local
//@ cdb-check:struct ref$<dyn$<associated_const_bindings::Trait<assoc$<N,101> > > > local = [...]

#![feature(gca_min_const_items)]
#![expect(unused_variables, incomplete_features)]

use std::gca;

trait Trait {
    #[rustc_always_gca]
    const N: usize;
}

impl Trait for () {
    const N: usize = gca!(101);
}

fn main() {
    let local = &() as &dyn Trait<N = 101>;

    zzz(); // #break
}

#[inline(never)]
fn zzz() {
    ()
}
