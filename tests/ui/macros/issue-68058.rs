//@ check-pass

macro_rules! foo {
    ($doc: expr) => {
        fn f() {
            #[doc = $doc]
            ()
        }
    };
}

foo!("doc");

fn main() {}
