// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr-test "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr-test "([0-9a-f][0-9a-f] |╾─*A(LLOC)?[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ normalize-stderr-test "HEX_DUMP\s*\n\s*HEX_DUMP" -> "HEX_DUMP"

const FOO: *const u32 = { //~ ERROR it is undefined behavior
    let x = 42;
    &x
};

union Union {
    ptr: *const u32
}

const BAR: Union = { //~ ERROR it is undefined behavior
    let x = 42;
    Union { ptr: &x }
};

const BAZ: Union = { //~ ERROR it is undefined behavior
    let x = 42_u32;
    Union { ptr: &(&x as *const u32) as *const *const u32 as _ }
};

fn main() {
    let x = FOO;
    let x = BAR;
}
