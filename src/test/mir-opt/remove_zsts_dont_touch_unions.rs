// compile-flags: -Zmir-opt-level=3

// Ensure RemoveZsts doesn't remove ZST assignments to union fields,
// which causes problems in Miri.

union Foo {
    x: (),
    y: u64,
}

// EMIT_MIR remove_zsts_dont_touch_unions.get_union.RemoveZsts.after.mir
fn get_union() -> Foo {
    Foo { x: () }
}


fn main() {
    get_union();
}
