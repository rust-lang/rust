pub mod m {
    pub struct S(u8);

    pub mod n {
        pub(in m) struct Z(pub(in m::n) u8);
    }
}

pub use m::S;
