//@ run-fail
//@ check-run-results

fn main() {
    let mut a = 1;
    a /= &0;
}
