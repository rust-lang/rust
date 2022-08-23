// check-pass
// compile-flags: -Zextra-const-ub-checks

#[derive(PartialEq, Eq, Copy, Clone)]
#[repr(packed)]
struct Foo {
    field: (i64, u32, u32, u32),
}

const FOO: Foo = Foo {
    field: (5, 6, 7, 8),
};

fn main() {
    match FOO {
        Foo { field: (5, 6, 7, 8) } => {},
        FOO => unreachable!(),
        _ => unreachable!(),
    }
}
