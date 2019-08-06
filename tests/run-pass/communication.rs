// ignore-windows: TODO env var emulation stubbed out on Windows
// compile-flags: -Zmiri-enable-communication

fn main() {
    assert!(std::env::var("PWD").is_ok());
}
