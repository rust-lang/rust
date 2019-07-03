// revisions: cfail1 cfail2 cfail3
// compile-flags: -Coverflow-checks=on
// build-pass (FIXME(62277): could be check-pass?)

#![allow(warnings)]
#![warn(const_err)]

fn main() {
    255u8 + 1; //~ WARNING this expression will panic at run-time
}
