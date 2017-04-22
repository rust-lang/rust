// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(conservative_impl_trait)]
#![feature(decl_macro)]

mod m {
    fn priv_fn() {}
    enum PrivEnum { Variant }
    pub enum PubEnum { Variant }
    trait PrivTrait { fn method() {} }
    impl PrivTrait for u8 {}
    pub trait PubTrait { fn method() {} }
    impl PubTrait for u8 {}
    struct PrivTupleStruct(u8);
    pub struct PubTupleStruct(u8);
    impl PubTupleStruct { fn method() {} }

    struct Priv;
    pub type Alias = Priv;
    pub struct Pub<T = Alias>(pub T);

    impl Pub<Priv> {
        pub fn static_method() {}
    }

    pub macro m() {
        priv_fn; //~ ERROR type `fn() {m::priv_fn}` is private
        PrivEnum::Variant; //~ ERROR type `m::PrivEnum` is private
        PubEnum::Variant; // OK
        <u8 as PrivTrait>::method; //~ ERROR type `fn() {<u8 as m::PrivTrait>::method}` is private
        <u8 as PubTrait>::method; // OK
        PrivTupleStruct;
        //~^ ERROR type `fn(u8) -> m::PrivTupleStruct {m::PrivTupleStruct::{{constructor}}}` is priv
        PubTupleStruct;
        //~^ ERROR type `fn(u8) -> m::PubTupleStruct {m::PubTupleStruct::{{constructor}}}` is privat
        Pub::static_method; //~ ERROR type `m::Priv` is private
    }

    trait Trait {}
    pub trait TraitWithTyParam<T> {}
    pub trait TraitWithAssocTy { type X; }
    impl Trait for u8 {}
    impl<T> TraitWithTyParam<T> for u8 {}
    impl TraitWithAssocTy for u8 { type X = Priv; }

    pub fn leak_anon1() -> impl Trait + 'static { 0 }
    pub fn leak_anon2() -> impl TraitWithTyParam<Alias> { 0 }
    pub fn leak_anon3() -> impl TraitWithAssocTy<X = Alias> { 0 }

    pub fn leak_dyn1() -> Box<Trait + 'static> { Box::new(0) }
    pub fn leak_dyn2() -> Box<TraitWithTyParam<Alias>> { Box::new(0) }
    pub fn leak_dyn3() -> Box<TraitWithAssocTy<X = Alias>> { Box::new(0) }
}

fn main() {
    m::Alias {}; //~ ERROR type `m::Priv` is private
    m::Pub { 0: m::Alias {} }; //~ ERROR type `m::Priv` is private
    m::m!();

    m::leak_anon1(); //~ ERROR trait `m::Trait` is private
    m::leak_anon2(); //~ ERROR type `m::Priv` is private
    m::leak_anon3(); //~ ERROR type `m::Priv` is private

    m::leak_dyn1(); //~ ERROR type `m::Trait + 'static` is private
    m::leak_dyn2(); //~ ERROR type `m::Priv` is private
    m::leak_dyn3(); //~ ERROR type `m::Priv` is private

    // Check that messages are not duplicated for various kinds of assignments
    let a = m::Alias {}; //~ ERROR type `m::Priv` is private
    let mut b = a; //~ ERROR type `m::Priv` is private
    b = a; //~ ERROR type `m::Priv` is private
    match a { //~ ERROR type `m::Priv` is private
        _ => {}
    }
}
