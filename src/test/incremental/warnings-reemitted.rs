// revisions: cfail1 cfail2 cfail3
// compile-flags: -Coverflow-checks=on
// build-pass (FIXME(62277): could be check-pass?)

#![warn(const_err)]

fn main() {
    let _ = 255u8 + 1; //~ WARNING attempt to add with overflow
}
