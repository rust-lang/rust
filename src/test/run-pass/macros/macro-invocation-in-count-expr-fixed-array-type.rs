// run-pass
// pretty-expanded FIXME #23616

macro_rules! four {
    () => (4)
}

fn main() {
    let _x: [u16; four!()];
}
