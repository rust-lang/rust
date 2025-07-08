//@ run-fail
//@ check-run-results

fn main() {
    let mut a: u8 = 0;
    a -= &1;
}
