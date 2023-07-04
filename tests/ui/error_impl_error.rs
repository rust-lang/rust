#![allow(unused)]
#![warn(clippy::error_impl_error)]
#![no_main]

mod a {
    #[derive(Debug)]
    struct Error;

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

mod c {
    union Error {
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

mod d {
    type Error = std::fmt::Error;
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
