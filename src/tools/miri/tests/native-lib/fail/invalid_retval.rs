// Only works on Unix targets
//@ignore-target: windows wasm
//@only-on-host
//@normalize-stderr-test: "OS `.*`" -> "$$OS"

extern "C" {
    fn u8_id(x: u8) -> bool;
}

fn main() {
    unsafe {
        u8_id(2); //~ ERROR: invalid value: encountered 0x02, but expected a boolean
    }
}
