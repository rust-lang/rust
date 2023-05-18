#![allow(unused)]
#![allow(private_in_public)]
#![deny(unnameable_types)]

mod m {
    pub struct PubStruct(pub i32); //~ ERROR struct `PubStruct` is reachable but cannot be named

    pub enum PubE { //~ ERROR enum `PubE` is reachable but cannot be named
        V(i32),
    }

    pub trait PubTr { //~ ERROR trait `PubTr` is reachable but cannot be named
        const C : i32 = 0;
        type Alias; //~ ERROR associated type `PubTr::Alias` is reachable but cannot be named
        fn f() {}
    }

    impl PubTr for PubStruct {
        type Alias = i32; //~ ERROR associated type `<PubStruct as PubTr>::Alias` is reachable but cannot be named
        fn f() {}
    }
}

pub trait Voldemort<T> {}

impl Voldemort<m::PubStruct> for i32 {}
impl Voldemort<m::PubE> for i32 {}
impl<T> Voldemort<T> for u32 where T: m::PubTr {}

fn main() {}
