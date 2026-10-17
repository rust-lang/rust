#[no_mangle]
static FOO: u8 = 1;
//~^ HELP: it's first defined here, in crate `clashing`

#[export_name = "FOO"]
static BAR: u8 = 2;
//~^ HELP: then it's defined here again, in crate `clashing`

fn main() {
    extern "Rust" {
        static FOO: u8;
    }
    let _val = &raw const FOO;
    //~^ ERROR: multiple definitions of symbol `FOO`
}
