// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr-test: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr-test: "([0-9a-f][0-9a-f] |╾─*A(LLOC)?[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ normalize-stderr-test: "HEX_DUMP\s*\n\s*HEX_DUMP" -> "HEX_DUMP"

pub struct Wrapper;
pub static MAGIC_FFI_REF: &'static Wrapper = unsafe {
//~^ERROR: it is undefined behavior to use this value
    std::mem::transmute(&{
        let y = 42;
        y
    })
};

fn main() {}
