

fn main() {
    let i: int = if false { fail } else { 5 };
    log_full(core::debug, i);
}
