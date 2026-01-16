// We need this feature as it changes `dylib` linking behavior and allows us to link to `rustc_driver`.
#![feature(rustc_private)]
#![cfg_attr(not(bootstrap), feature(on_broken_pipe))]

#[cfg_attr(not(bootstrap), std::io::on_broken_pipe)]
#[cfg(not(bootstrap))]
fn on_broken_pipe() -> std::io::OnBrokenPipe {
    // FIXME(#131436): Ideally there would be no use of e.g. `println!()` in the compiler.
    std::io::OnBrokenPipe::Kill
}

fn main() {
    rustdoc::main()
}
