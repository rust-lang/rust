// run-pass
// check-run-results

fn main() {
    println!(include!("<leading-lt"));
    println!(include!("trailing-gt>"));
}
