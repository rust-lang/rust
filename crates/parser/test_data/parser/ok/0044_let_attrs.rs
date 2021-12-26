// https://github.com/rust-analyzer/rust-analyzer/issues/677
fn main() {
    #[cfg(feature = "backtrace")]
    let exit_code = panic::catch_unwind(move || main());
}
