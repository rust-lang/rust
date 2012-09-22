
use baz::zed;
use zed::bar;

mod baz {
    #[legacy_exports];
    mod zed {
        #[legacy_exports];
        fn bar() { debug!("bar2"); }
    }
}

fn main() { bar(); }
