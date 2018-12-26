// error-pattern:internal error: entered unreachable code: 6 is not prime
fn main() {
    unreachable!("{} is not {}", 6u32, "prime");
}
