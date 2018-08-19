// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(decl_macro, associated_type_defaults)]
#![allow(unused, private_in_public)]

mod priv_trait {
    trait PrivTr {
        type AssocTy = u8;
    }
    pub trait PubTr: PrivTr {}

    pub macro mac1() {
        let _: Box<PubTr<AssocTy = u8>>;
        //~^ ERROR type `(dyn priv_trait::PubTr<AssocTy=u8> + '<empty>)` is private
        //~| ERROR type `(dyn priv_trait::PubTr<AssocTy=u8> + '<empty>)` is private
        type InSignatureTy2 = Box<PubTr<AssocTy = u8>>;
        //~^ ERROR type `(dyn priv_trait::PubTr<AssocTy=u8> + 'static)` is private
        trait InSignatureTr2: PubTr<AssocTy = u8> {}
        //~^ ERROR trait `priv_trait::PrivTr` is private
    }
    pub macro mac2() {
        let _: Box<PrivTr<AssocTy = u8>>;
        //~^ ERROR type `(dyn priv_trait::PrivTr<AssocTy=u8> + '<empty>)` is private
        //~| ERROR type `(dyn priv_trait::PrivTr<AssocTy=u8> + '<empty>)` is private
        type InSignatureTy1 = Box<PrivTr<AssocTy = u8>>;
        //~^ ERROR type `(dyn priv_trait::PrivTr<AssocTy=u8> + 'static)` is private
        trait InSignatureTr1: PrivTr<AssocTy = u8> {}
        //~^ ERROR trait `priv_trait::PrivTr` is private
    }
}
fn priv_trait1() {
    priv_trait::mac1!();
}
fn priv_trait2() {
    priv_trait::mac2!();
}

mod priv_parent_substs {
    pub trait PubTrWithParam<T = Priv> {
        type AssocTy = u8;
    }
    struct Priv;
    pub trait PubTr: PubTrWithParam<Priv> {}

    pub macro mac() {
        let _: Box<PubTrWithParam<AssocTy = u8>>;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        //~| ERROR type `priv_parent_substs::Priv` is private
        let _: Box<PubTr<AssocTy = u8>>;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        //~| ERROR type `priv_parent_substs::Priv` is private
        pub type InSignatureTy1 = Box<PubTrWithParam<AssocTy = u8>>;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        pub type InSignatureTy2 = Box<PubTr<AssocTy = u8>>;
        //~^ ERROR type `priv_parent_substs::Priv` is private
        trait InSignatureTr1: PubTrWithParam<AssocTy = u8> {}
        //~^ ERROR type `priv_parent_substs::Priv` is private
        trait InSignatureTr2: PubTr<AssocTy = u8> {}
        //~^ ERROR type `priv_parent_substs::Priv` is private
    }
}
fn priv_parent_substs() {
    priv_parent_substs::mac!();
}

fn main() {}
