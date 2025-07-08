//@ run-fail
//@ check-run-results

fn main() {
    let _ = 1 % &0;
}
