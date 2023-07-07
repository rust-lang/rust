//@revisions: allow_private disallow_private
//@[allow_private] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/error_impl_error/allow_private
//@[disallow_private] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/error_impl_error/disallow_private
#![allow(unused)]
#![warn(clippy::error_impl_error)]
#![no_main]

pub mod a {
    #[derive(Debug)]
    pub struct Error;

    impl std::fmt::Display for Error {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl std::error::Error for Error {}
}

mod b {
    #[derive(Debug)]
    enum Error {}

    impl std::fmt::Display for Error {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl std::error::Error for Error {}
}

pub mod c {
    pub union Error {
        a: u32,
        b: u32,
    }

    impl std::fmt::Debug for Error {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl std::fmt::Display for Error {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl std::error::Error for Error {}
}

pub mod d {
    pub type Error = std::fmt::Error;
}

mod e {
    #[derive(Debug)]
    struct MyError;

    impl std::fmt::Display for MyError {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl std::error::Error for MyError {}
}

mod f {
    type MyError = std::fmt::Error;
}
