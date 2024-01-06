// normalize-stderr-test "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
// normalize-stderr-test "([0-9a-f][0-9a-f] |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
#![feature(const_refs_to_static)]

static S: i8 = 10;

const C: &bool = unsafe { std::mem::transmute(&S) };
//~^ERROR: undefined behavior
//~| expected a boolean

fn main() {
    // This must be rejected here (or earlier), since it's not a valid `&bool`.
    match &true {
        C => {}, //~ERROR: could not evaluate constant pattern
        _ => {},
    }
}
