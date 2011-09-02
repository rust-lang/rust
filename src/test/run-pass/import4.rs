
import zed::bar;

mod zed {
    fn bar() { log "bar"; }
}

fn main(args: [istr]) { let zed = 42; bar(); }
