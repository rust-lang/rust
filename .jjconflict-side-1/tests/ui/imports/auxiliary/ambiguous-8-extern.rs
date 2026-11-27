mod t2 {
    #[derive(Debug)]
    pub enum Error {}

    mod t {
        pub trait Error: Sized {}
    }

    use self::t::*;
}

pub use t2::*;
