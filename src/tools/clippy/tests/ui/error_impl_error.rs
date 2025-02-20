#![allow(unused)]
#![warn(clippy::error_impl_error)]
#![no_main]

pub mod a {
    #[derive(Debug)]
    pub struct Error;
    //~^ error_impl_error

    impl std::fmt::Display for Error {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl std::error::Error for Error {}
}

mod b {
    #[derive(Debug)]
    pub(super) enum Error {}
    //~^ error_impl_error

    impl std::fmt::Display for Error {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl std::error::Error for Error {}
}

pub mod c {
    pub union Error {
        //~^ error_impl_error
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
    //~^ error_impl_error
}

mod e {
    #[derive(Debug)]
    pub(super) struct MyError;

    impl std::fmt::Display for MyError {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl std::error::Error for MyError {}
}

pub mod f {
    pub type MyError = std::fmt::Error;
}

// Do not lint module-private types

mod g {
    #[derive(Debug)]
    enum Error {}

    impl std::fmt::Display for Error {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl std::error::Error for Error {}
}

mod h {
    type Error = std::fmt::Error;
}
