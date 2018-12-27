#![feature(box_syntax)]



fn borrow(_v: &isize) {}

fn local() {
    let mut v: Box<_> = box 3;
    borrow(&*v);
}

fn local_rec() {
    struct F { f: Box<isize> }
    let mut v = F {f: box 3};
    borrow(&*v.f);
}

fn local_recs() {
    struct F { f: G }
    struct G { g: H }
    struct H { h: Box<isize> }
    let mut v = F {f: G {g: H {h: box 3}}};
    borrow(&*v.f.g.h);
}

fn aliased_imm() {
    let mut v: Box<_> = box 3;
    let w = &v;
    borrow(&*v);
    w.use_ref();
}

fn aliased_mut() {
    let mut v: Box<_> = box 3;
    let w = &mut v;
    borrow(&*v); //~ ERROR cannot borrow `*v`
    w.use_mut();
}

fn aliased_other() {
    let mut v: Box<_> = box 3;
    let mut w: Box<_> = box 4;
    let x = &mut w;
    borrow(&*v);
    x.use_mut();
}

fn aliased_other_reassign() {
    let mut v: Box<_> = box 3;
    let mut w: Box<_> = box 4;
    let mut x = &mut w;
    x = &mut v;
    borrow(&*v); //~ ERROR cannot borrow `*v`
    x.use_mut();
}

fn main() {
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
