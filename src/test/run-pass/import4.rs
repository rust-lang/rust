
import zed::bar;

mod zed {
    fn bar() { log "bar"; }
}

fn main(args: [str]) { let zed = 42; bar(); }