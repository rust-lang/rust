//@ run-fail
//@ check-run-results

fn main() {
    let _: u8 = 255 * &2;
}
