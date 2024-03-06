//@ run-pass
//@ pretty-expanded FIXME #23616

pub fn main() {
    const FOO: f64 = 10.0;

    match 0.0 {
        0.0 ..= FOO => (),
        _ => ()
    }
}
