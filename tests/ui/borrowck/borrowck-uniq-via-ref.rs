//@ run-pass
#![allow(dead_code)]


struct Rec {
    f: Box<isize>,
}

struct Outer {
    f: Inner
}

struct Inner {
    g: Innermost
}

struct Innermost {
    h: Box<isize>,
}

fn borrow(_v: &isize) {}

fn box_mut(v: &mut Box<isize>) {
    borrow(&**v); // OK: &mut -> &imm
}

fn box_mut_rec(v: &mut Rec) {
    borrow(&*v.f); // OK: &mut -> &imm
}

fn box_mut_recs(v: &mut Outer) {
    borrow(&*v.f.g.h); // OK: &mut -> &imm
}

fn box_imm(v: &Box<isize>) {
    borrow(&**v); // OK
}

fn box_imm_rec(v: &Rec) {
    borrow(&*v.f); // OK
}

fn box_imm_recs(v: &Outer) {
    borrow(&*v.f.g.h); // OK
}

pub fn main() {
}
