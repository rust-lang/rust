// compile-flags: -C overflow-checks=no

fn use_zst(_: ((), ())) {}

struct Temp {
    x: u8,
}

fn use_u8(_: u8) {}

// EMIT_MIR rustc.main.SimplifyLocals.diff
fn main() {
    let ((), ()) = ((), ());
    use_zst(((), ()));

    use_u8((Temp { x: 40 }).x + 2);
}
