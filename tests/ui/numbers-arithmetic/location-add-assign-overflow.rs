//@ run-fail
//@ check-run-results

fn main() {
    let mut a: u8 = 255;
    a += &1;
}
