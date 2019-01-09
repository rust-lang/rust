// gate-test-log_syntax

fn main() {
    println!("{:?}", log_syntax!()); //~ ERROR `log_syntax!` is not stable
}
