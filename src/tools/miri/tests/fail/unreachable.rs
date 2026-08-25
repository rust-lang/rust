fn main() {
    unsafe { std::hint::unreachable_unchecked() } //~ERROR: entering unreachable code
}
