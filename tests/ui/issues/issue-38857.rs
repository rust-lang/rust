fn main() {
    let a = std::sys::imp::process::process_common::StdioPipes { ..panic!() };
    //~^ ERROR cannot find item `imp`
    //~^^ ERROR module `sys` is private [E0603]
}
