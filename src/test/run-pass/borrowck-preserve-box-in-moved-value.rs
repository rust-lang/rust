// exec-env:RUST_POISON_ON_FREE=1

// Test that we root `x` even though it is found in immutable memory,
// because it is moved.

#[feature(managed_boxes)];

fn free<T>(x: @T) {}

struct Foo {
    f: @Bar
}

struct Bar {
    g: int
}

fn lend(x: @Foo) -> int {
    let y = &x.f.g;
    free(x); // specifically here, if x is not rooted, it will be freed
    *y
}

pub fn main() {
    assert_eq!(lend(@Foo {f: @Bar {g: 22}}), 22);
}
