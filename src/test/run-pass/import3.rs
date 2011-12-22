
import baz::zed;
import zed::bar;

mod baz {
    mod zed {
        fn bar() { #debug("bar2"); }
    }
}

fn main() { bar(); }
