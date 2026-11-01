pub mod m {
    pub struct S(u8);

    pub mod n {
        pub(in crate::m) struct Z(pub(in crate::m::n) u8);
    }
}

pub use m::S;
