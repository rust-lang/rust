// run-pass
// pretty-expanded FIXME(#23616)

// FIXME(#41620)
#![allow(illegal_floating_point_literal_pattern)]

pub fn main() {
    const FOO: f64 = 10.0;

    match 0.0 {
        0.0 ..= FOO => (),
        _ => ()
    }
}
