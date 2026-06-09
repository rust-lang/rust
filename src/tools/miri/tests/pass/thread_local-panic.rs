thread_local! {
    static LOCAL: u64 = panic!();

}

fn main() {
    let _ = std::panic::catch_unwind(|| LOCAL.with(|_| {}));
}
