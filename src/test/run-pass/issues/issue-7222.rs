// run-pass
// pretty-expanded FIXME #23616
#![allow(illegal_floating_point_literal_pattern)] // FIXME #41620

pub fn main() {
    const FOO: f64 = 10.0;

    match 0.0 {
        0.0 ..= FOO => (),
        _ => ()
    }
}
