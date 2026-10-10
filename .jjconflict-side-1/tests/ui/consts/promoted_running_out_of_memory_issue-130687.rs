//! Ensure we do not ICE when a promoted fails to evaluate due to running out of memory.
//! Also see <https://github.com/rust-lang/rust/issues/130687>.

// Needs the max type size to be much bigger than the RAM people typically have.
//@ only-64bit
// FIXME (#135952) In some cases on AArch64 Linux the diagnostic does not trigger
//@ ignore-aarch64-unknown-linux-gnu
// AIX will allow the allocation to go through, and get SIGKILL when zero initializing
// the overcommitted page.
//@ ignore-aix

pub struct Data([u8; (1 << 47) - 1]);
const _: &'static Data = &Data([0; (1 << 47) - 1]);
//~^ ERROR: tried to allocate more memory than available to compiler

fn main() {}
