
import zed::bar;
import baz::zed;

mod baz {
    mod zed {
        fn bar() { log "bar2"; }
    }
}

fn main(vec[str] args) { bar(); }