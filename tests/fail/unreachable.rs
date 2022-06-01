// error-pattern: entering unreachable code
fn main() {
    unsafe { std::hint::unreachable_unchecked() }
}
