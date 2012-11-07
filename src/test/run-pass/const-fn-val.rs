fn foo() -> int {
    return 0xca7f000d;
}

struct Bar { f: &fn() -> int }

const b : Bar = Bar { f: foo };

fn main() {
    assert (b.f)() == 0xca7f000d;
}