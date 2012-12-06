extern mod std;

use std::getopts::*;

fn main() {
    let args = ~[];
    let opts = ~[optopt(~"b")];

    match getopts(args, opts) {
        result::Ok(ref m)  =>
            assert !opt_present(m, "b"),
        result::Err(ref f) => fail fail_str(*f)
    };

}
