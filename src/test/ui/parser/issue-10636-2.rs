// FIXME(31528) we emit a bunch of silly errors here due to continuing past the
// first one. This would be easy-ish to address by better recovery in tokenisation.

// compile-flags: -Z parse-only

pub fn trace_option(option: Option<isize>) {
    option.map(|some| 42;
                          //~^ ERROR: expected one of

} //~ ERROR: incorrect close delimiter
//~^ ERROR: expected expression, found `)`

fn main() {}
