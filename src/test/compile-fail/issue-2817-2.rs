fn uuid() -> uint { fail; }

fn from_str(s: ~str) -> uint { fail; }
fn to_str(u: uint) -> ~str { fail; }
fn uuid_random() -> uint { fail; }

fn main() {
    for uint::range(0, 100000) |_i| { //~ ERROR A for-loop body must return (), but
        false
    }
}