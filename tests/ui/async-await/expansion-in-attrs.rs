//@ check-pass
//@ edition:2018

macro_rules! with_doc {
    ($doc: expr) => {
        #[doc = $doc]
        async fn f() {}
    };
}

with_doc!(concat!(""));

fn main() {}
