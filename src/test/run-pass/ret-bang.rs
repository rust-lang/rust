


// -*- rust -*-
fn my_err(str s) -> ! { log_err s; fail; }

fn okay(uint i) -> int {
    if (i == 3u) { my_err("I don't like three"); } else { ret 42; }
}

fn main() { okay(4u); }