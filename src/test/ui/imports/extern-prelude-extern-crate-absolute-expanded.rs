// build-pass (FIXME(62277): could be check-pass?)
// edition:2018

macro_rules! define_iso { () => {
    extern crate std as iso;
}}

::iso::thread_local! {
    static S: u8 = 0;
}

define_iso!();

fn main() {
    let s = S;
}
