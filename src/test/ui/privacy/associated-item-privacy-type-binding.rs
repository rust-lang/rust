#![feature(decl_macro, associated_type_defaults)]
#![allow(unused, private_in_public)]

mod priv_trait {
    trait PrivTr {
        type AssocTy = u8;
    }
    pub trait PubTr: PrivTr {}

    pub macro mac1() {
        let _: Box<PubTr<AssocTy = u8>>;
        //~^ ERROR trait `priv_trait::PrivTr` is private
        //~| ERROR trait `priv_trait::PrivTr` is private
        type InSignatureTy2 = Box<PubTr<AssocTy = u8>>;
        //~^ ERROR trait `priv_trait::PrivTr` is private
        trait InSignatureTr2: PubTr<AssocTy = u8> {}
        //~^ ERROR trait `priv_trait::PrivTr` is private
    }
    pub macro mac2() {
        let _: Box<PrivTr<AssocTy = u8>>;
        //~^ ERROR trait `priv_trait::PrivTr` is private
        //~| ERROR trait `priv_trait::PrivTr` is private
        type InSignatureTy1 = Box<PrivTr<AssocTy = u8>>;
        //~^ ERROR trait `priv_trait::PrivTr` is private
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
