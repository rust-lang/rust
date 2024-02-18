//@ build-pass

// Check that the evaluation of const-functions with
// zero-sized types as arguments compiles successfully

struct Zst {}

const fn foo(val: Zst) -> Zst { val }

const FOO: Zst = foo(Zst {});

fn main() {
    const _: Zst = FOO;
}
