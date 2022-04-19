// run-pass
// compile-flags: -C lto -C target-feature=+crt-static -C rpath=no -C opt-level=2
// no-prefer-dynamic

fn main() {
    let _ = std::process::Command::new("true").spawn();
}
