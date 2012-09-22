


// -*- rust -*-
mod inner {
    #[legacy_exports];
    mod inner2 {
        #[legacy_exports];
        fn hello() { debug!("hello, modular world"); }
    }
    fn hello() { inner2::hello(); }
}

fn main() { inner::hello(); inner::inner2::hello(); }
