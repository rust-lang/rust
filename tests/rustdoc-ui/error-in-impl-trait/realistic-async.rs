//@ edition:2018
//@ check-pass

mod windows {
    pub trait WinFoo {
        fn foo(&self) {}
    }

    impl WinFoo for () {}
}

#[cfg(any(windows, doc))]
use windows::*;

mod unix {
    pub trait UnixFoo {
        fn foo(&self) {}
    }

    impl UnixFoo for () {}
}

#[cfg(any(unix, doc))]
use unix::*;

async fn bar() {
    ().foo()
}
