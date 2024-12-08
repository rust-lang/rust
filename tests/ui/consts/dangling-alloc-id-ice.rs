// https://github.com/rust-lang/rust/issues/55223
// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr-test: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr-test: "([0-9a-f][0-9a-f] |╾─*A(LLOC)?[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ normalize-stderr-test: "HEX_DUMP\s*\n\s*HEX_DUMP" -> "HEX_DUMP"

union Foo<'a> {
    y: &'a (),
    long_live_the_unit: &'static (),
}

const FOO: &() = {
    //~^ ERROR it is undefined behavior to use this value
    let y = ();
    unsafe { Foo { y: &y }.long_live_the_unit }
};

fn main() {}
