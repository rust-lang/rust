
import baz::zed;
import zed::bar;

mod baz {
    mod zed {
        fn bar() { log "bar2"; }
    }
}

fn main() { bar(); }
