// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl

fn main() {
    let a = std::sys::imp::process::process_common::StdioPipes { ..panic!() };
    //~^ ERROR failed to resolve: could not find `imp` in `sys` [E0433]
    //~^^ ERROR module `sys` is private [E0603]
}
