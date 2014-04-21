// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that you can use an early-bound lifetime parameter as
// on of the generic parameters in a trait.


trait Trait<'a> {
    fn long(&'a self) -> int;
    fn short<'b>(&'b self) -> int;
}

fn poly_invoke<'c, T: Trait<'c>>(x: &'c T) -> (int, int) {
    let l = x.long();
    let s = x.short();
    (l,s)
}

fn object_invoke1<'d>(x: &'d Trait<'d>) -> (int, int) {
    let l = x.long();
    let s = x.short();
    (l,s)
}

struct Struct1<'e> {
    f: &'e Trait<'e>
}

fn field_invoke1<'f, 'g>(x: &'g Struct1<'f>) -> (int,int) {
    let l = x.f.long();
    let s = x.f.short();
    (l,s)
}

struct Struct2<'h, 'i> {
    f: &'h Trait<'i>
}

fn object_invoke2<'j, 'k>(x: &'k Trait<'j>) -> int {
    x.short()
}

fn field_invoke2<'l, 'm, 'n>(x: &'n Struct2<'l,'m>) -> int {
    x.f.short()
}

trait MakerTrait<'o> {
    fn mk() -> Self;
}

fn make_val<'p, T:MakerTrait<'p>>() -> T {
    MakerTrait::mk()
}

trait RefMakerTrait<'q> {
    fn mk(Self) -> &'q Self;
}

fn make_ref<'r, T:RefMakerTrait<'r>>(t:T) -> &'r T {
    RefMakerTrait::mk(t)
}

impl<'s> Trait<'s> for (int,int) {
    fn long(&'s self) -> int {
        let &(x,_) = self;
        x
    }
    fn short<'b>(&'b self) -> int {
        let &(_,y) = self;
        y
    }
}

impl<'t> MakerTrait<'t> for Box<Trait<'t>> {
    fn mk() -> Box<Trait<'t>> { box() (4i,5i) as Box<Trait> }
}

enum List<'l> {
    Cons(int, &'l List<'l>),
    Null
}

impl<'l> List<'l> {
    fn car<'m>(&'m self) -> int {
        match self {
            &Cons(car, _) => car,
            &Null => fail!(),
        }
    }
    fn cdr<'n>(&'n self) -> &'l List<'l> {
        match self {
            &Cons(_, cdr) => cdr,
            &Null => fail!(),
        }
    }
}

impl<'t> RefMakerTrait<'t> for List<'t> {
    fn mk(l:List<'t>) -> &'t List<'t> {
        l.cdr()
    }
}

pub fn main() {
    let t = (2i,3i);
    let o = &t as &Trait;
    let s1 = Struct1 { f: o };
    let s2 = Struct2 { f: o };
    assert_eq!(poly_invoke(&t), (2,3));
    assert_eq!(object_invoke1(&t), (2,3));
    assert_eq!(field_invoke1(&s1), (2,3));
    assert_eq!(object_invoke2(&t), 3);
    assert_eq!(field_invoke2(&s2), 3);

    let m : Box<Trait> = make_val();
    assert_eq!(object_invoke1(m), (4,5));
    assert_eq!(object_invoke2(m), 5);

    // The RefMakerTrait above is pretty strange (i.e. it is strange
    // to consume a value of type T and return a &T).  Easiest thing
    // that came to my mind: consume a cell of a linked list and
    // return a reference to the list it points to.
    let l0 = Null;
    let l1 = Cons(1, &l0);
    let l2 = Cons(2, &l1);
    let rl1 = &l1;
    let r  = make_ref(l2);
    assert_eq!(rl1.car(), r.car());
}
