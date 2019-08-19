// build-pass (FIXME(62277): could be check-pass?)
macro_rules! a {
    () => { "a" }
}

macro_rules! b {
    ($doc:expr) => {
        #[doc = $doc]
        pub struct B;
    }
}

b!(a!());

fn main() {}
