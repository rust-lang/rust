
import zed::bar;

mod zed {
    fn bar() { #debug("bar"); }
}

fn main(args: [str]) { let zed = 42; bar(); }
