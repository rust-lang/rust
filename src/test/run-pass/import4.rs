
import zed::bar;

mod zed {
    fn bar() { log "bar"; }
}

fn main(args: vec[str]) { let zed = 42; bar(); }