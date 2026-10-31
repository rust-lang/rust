//@ test-mir-pass: RemoveZsts

union Foo {
    x: (),
    y: u64,
}

// EMIT_MIR remove_zsts.get_union.RemoveZsts.diff
fn get_union() -> Foo {
    // CHECK-LABEL: fn get_union
    // CHECK: _0 = Foo { x: const () };
    Foo { x: () }
}

const MYSTERY: usize = 280_usize.isqrt() - 260_usize.isqrt();

// EMIT_MIR remove_zsts.remove_generic_array.RemoveZsts.diff
fn remove_generic_array<T: Copy>(x: T) -> ([T; 0], [T; MYSTERY]) {
    // CHECK-LABEL: fn remove_generic_array
    // CHECK: debug a => const ZeroSized: [T; 0];
    // CHECK: debug b => const ZeroSized: [T; 0];
    // CHECK: _0 = (const ZeroSized: [T; 0], const ZeroSized: [T; 0]);
    let a = [x; 0];
    let b = [x; MYSTERY];
    (a, b)
}

fn main() {
    get_union();
}
