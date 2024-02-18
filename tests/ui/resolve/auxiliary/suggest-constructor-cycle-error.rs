mod m {
    pub struct Uuid(());

    impl Uuid {
        pub fn encode_buffer() -> [u8; LENGTH] {
            []
        }
    }
    const LENGTH: usize = 0;
}

pub use m::Uuid;
