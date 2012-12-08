fn uuid() -> uint { fail; }

fn from_str(s: ~str) -> uint { fail; }
fn to_str(u: uint) -> ~str { fail; }
fn uuid_random() -> uint { fail; }

fn main() {
    do uint::range(0, 100000) |_i| { //~ ERROR Do-block body must return bool, but
    }
    // should get a more general message if the callback
    // doesn't return nil
    do uint::range(0, 100000) |_i| { //~ ERROR mismatched types
        ~"str"
    }
}