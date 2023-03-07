//@compile-flags: -Zmiri-disable-isolation

fn getpid() -> u32 {
    std::process::id()
}

fn main() {
    getpid();
}
