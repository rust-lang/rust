#![feature(link_args)]

#[allow(unused_attributes)]
// Set the stack size at link time on Windows. See rustc_driver::in_rustc_thread
// for the rationale.
#[cfg_attr(all(windows, target_env = "msvc"), link_args = "/STACK:16777216")]
// We only build for msvc and gnu now, but we use a exhaustive condition here
// so we can expect either the stack size to be set or the build fails.
#[cfg_attr(all(windows, not(target_env = "msvc")), link_args = "-Wl,--stack,16777216")]
// See src/rustc/rustc.rs for the corresponding rustc settings.
extern {}

extern crate rustdoc;

fn main() { rustdoc::main() }
