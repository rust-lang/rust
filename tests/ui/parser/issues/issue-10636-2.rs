//@ error-pattern: mismatched closing delimiter: `}`
// FIXME(31528) we emit a bunch of silly errors here due to continuing past the
// first one. This would be easy-ish to address by better recovery in tokenisation.

pub fn trace_option(option: Option<isize>) {
    option.map(|some| 42;


}

fn main() {}
