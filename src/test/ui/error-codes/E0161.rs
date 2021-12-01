// Check that E0161 is a hard error in all possible configurations that might
// affect it.

// revisions: migrate nll zflags edition migrateul nllul zflagsul editionul
//[zflags]compile-flags: -Z borrowck=migrate
//[edition]edition:2018
//[zflagsul]compile-flags: -Z borrowck=migrate
//[editionul]edition:2018
//[migrateul] check-pass
//[nllul] check-pass
//[zflagsul] check-pass
//[editionul] check-pass

// Since we are testing nll (and migration) explicitly as a separate
// revisions, don't worry about the --compare-mode=nll on this test.

// ignore-compare-mode-nll

#![allow(incomplete_features)]
#![cfg_attr(nll, feature(nll))]
#![cfg_attr(nllul, feature(nll))]
#![cfg_attr(migrateul, feature(unsized_locals))]
#![cfg_attr(zflagsul, feature(unsized_locals))]
#![cfg_attr(nllul, feature(unsized_locals))]
#![cfg_attr(editionul, feature(unsized_locals))]

trait Bar {
    fn f(self);
}

fn foo(x: Box<dyn Bar>) {
    x.f();
    //[migrate,nll,zflags,edition]~^ ERROR E0161
}

fn main() {}
