fn main() {
    let a = std::sys::imp::process::process_common::StdioPipes { ..panic!() };
    //~^ ERROR failed to resolve. Could not find `imp` in `sys` [E0433]
    //~^^ ERROR module `sys` is private [E0603]
}
