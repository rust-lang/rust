// build-pass
// edition:2018
// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir

#![feature(generators)]

fn main() {
    let _ = static |x: u8| match x {
        y if { yield } == y + 1 => (),
        _ => (),
    };
}
