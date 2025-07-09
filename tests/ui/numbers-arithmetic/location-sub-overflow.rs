//@ run-fail
//@ check-run-results

fn main() {
    let _: u8 = 0 - &1;
}
