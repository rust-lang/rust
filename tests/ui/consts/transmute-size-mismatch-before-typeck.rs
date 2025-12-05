//@ normalize-stderr-64bit: "8-byte" -> "word size"
//@ normalize-stderr-32bit: "4-byte" -> "word size"
//@ normalize-stderr-64bit: "64 bits" -> "word size"
//@ normalize-stderr-32bit: "32 bits" -> "word size"
//@ normalize-stderr-64bit: "16-byte" -> "2 * word size"
//@ normalize-stderr-32bit: "8-byte" -> "2 * word size"
//@ normalize-stderr-64bit: "128 bits" -> "2 * word size"
//@ normalize-stderr-32bit: "64 bits" -> "2 * word size"

fn main() {
    match &b""[..] {
        ZST => {} // ok, `const` error already emitted
    }
}

const ZST: &[u8] = unsafe { std::mem::transmute(1usize) };
//~^ ERROR transmuting from
//~| ERROR cannot transmute between types of different sizes
