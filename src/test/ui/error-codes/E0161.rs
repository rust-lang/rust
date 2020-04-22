// ignore-compare-mode-nll

// Check that E0161 is a hard error in all possible configurations that might
// affect it.

// revisions: migrate nll zflags edition migrateul nllul zflagsul editionul
//[zflags]compile-flags: -Z borrowck=migrate
//[edition]edition:2018
//[zflagsul]compile-flags: -Z borrowck=migrate
//[editionul]edition:2018

#![allow(incomplete_features)]
#![cfg_attr(nll, feature(nll))]
#![cfg_attr(nllul, feature(nll))]
#![cfg_attr(migrateul, feature(unsized_locals))]
#![cfg_attr(zflagsul, feature(unsized_locals))]
#![cfg_attr(nllul, feature(unsized_locals))]
#![cfg_attr(editionul, feature(unsized_locals))]
#![feature(box_syntax)]

fn foo(x: Box<[i32]>) {
    box *x;
    //[migrate,nll,zflags,edition]~^ ERROR E0161
    //[migrateul,nllul,zflagsul,editionul]~^^ ERROR E0161
}

fn main() {}
