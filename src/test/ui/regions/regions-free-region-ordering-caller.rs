// Test various ways to construct a pointer with a longer lifetime
// than the thing it points at and ensure that they result in
// errors. See also regions-free-region-ordering-callee.rs

// revisions: migrate nll
//[nll]compile-flags: -Z borrowck=mir

// Since we are testing nll (and migration) explicitly as a separate
// revisions, don't worry about the --compare-mode=nll on this test.

// ignore-compare-mode-nll

struct Paramd<'a> { x: &'a usize }

fn call2<'a, 'b>(a: &'a usize, b: &'b usize) {
    let z: Option<&'b &'a usize> = None;//[migrate]~ ERROR E0491
    //[nll]~^ ERROR lifetime may not live long enough
}

fn call3<'a, 'b>(a: &'a usize, b: &'b usize) {
    let y: Paramd<'a> = Paramd { x: a };
    let z: Option<&'b Paramd<'a>> = None;//[migrate]~ ERROR E0491
    //[nll]~^ ERROR lifetime may not live long enough
}

fn call4<'a, 'b>(a: &'a usize, b: &'b usize) {
    let z: Option<&'a &'b usize> = None;//[migrate]~ ERROR E0491
    //[nll]~^ ERROR lifetime may not live long enough
}

fn main() {}
