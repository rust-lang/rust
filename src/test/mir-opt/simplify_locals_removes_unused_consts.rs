// unit-test: SimplifyLocals
// compile-flags: -C overflow-checks=no

fn use_zst(_: ((), ())) {}

struct Temp {
    x: u8,
}

fn use_u8(_: u8) {}

// EMIT_MIR simplify_locals_removes_unused_consts.main.SimplifyLocals.diff
fn main() {
    let ((), ()) = ((), ());
    use_zst(((), ()));

    use_u8((Temp { x: 40 }).x + 2);
}
