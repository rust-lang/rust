extern "C" {
    pub static symbol: u32;
}
static CRASH: u32 = symbol;
//~^ ERROR use of extern static is unsafe and requires
//~| ERROR could not evaluate static initializer

fn main() {}
