// Test for #56254, we previously allowed the last example on the 2018
// editiion. Make sure that we now emit a warning in that case and an error for
// everyone else.

//ignore-compare-mode-nll

//revisions: ast migrate2015 migrate2018 nll2015 nll2018

//[migrate2015] compile-flags: -Zborrowck=migrate -Ztwo-phase-borrows
//[migrate2018] edition:2018
//[nll2018] edition:2018

#![cfg_attr(any(nll2015, nll2018), feature(nll))]

fn double_conflicts() {
    let mut v = vec![0, 1, 2];
    let shared = &v;

    v.extend(shared);
    //[migrate2015]~^ ERROR cannot borrow `v` as mutable
    //[nll2015]~^^ ERROR cannot borrow `v` as mutable
    //[migrate2018]~^^^ ERROR cannot borrow `v` as mutable
    //[nll2018]~^^^^ ERROR cannot borrow `v` as mutable
    //[ast]~^^^^^ ERROR cannot borrow `v` as mutable
}

fn activation_conflict() {
    let mut v = vec![0, 1, 2];

    v.extend(&v);
    //[migrate2015]~^ ERROR cannot borrow `v` as mutable
    //[nll2015]~^^ ERROR cannot borrow `v` as mutable
    //[migrate2018]~^^^ ERROR cannot borrow `v` as mutable
    //[nll2018]~^^^^ ERROR cannot borrow `v` as mutable
    //[ast]~^^^^^ ERROR cannot borrow `v` as immutable
}

fn reservation_conflict() {
    let mut v = vec![0, 1, 2];
    let shared = &v;

    v.push(shared.len());
    //[nll2015]~^ ERROR cannot borrow `v` as mutable
    //[nll2018]~^^ ERROR cannot borrow `v` as mutable
    //[migrate2015]~^^^ WARNING cannot borrow `v` as mutable
    //[migrate2015]~| WARNING may become a hard error in the future

    //[migrate2018]~^^^^^^ WARNING cannot borrow `v` as mutable
    //[migrate2018]~| WARNING may become a hard error in the future

    //[ast]~^^^^^^^^^ ERROR cannot borrow `v` as mutable
}

fn main() {}
