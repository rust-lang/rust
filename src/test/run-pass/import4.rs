
import zed::bar;

mod zed {
    fn bar() { log "bar"; }
}

fn main(vec[str] args) { auto zed = 42; bar(); }