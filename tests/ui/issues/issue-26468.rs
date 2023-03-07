// run-pass
#![allow(dead_code)]

enum FooMode {
    Check = 0x1001,
}

enum BarMode {
    Check = 0x2001,
}

enum Mode {
    Foo(FooMode),
    Bar(BarMode),
}

#[inline(never)]
fn broken(mode: &Mode) -> u32 {
    for _ in 0..1 {
        if let Mode::Foo(FooMode::Check) = *mode { return 17 }
        if let Mode::Bar(BarMode::Check) = *mode { return 19 }
    }
    return 42;
}

fn main() {
    let mode = Mode::Bar(BarMode::Check);
    assert_eq!(broken(&mode), 19);
}
