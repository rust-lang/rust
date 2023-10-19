// skip-filecheck
union Foo {
    x: (),
    y: u64,
}

// EMIT_MIR remove_zsts.get_union.RemoveZsts.diff
// EMIT_MIR remove_zsts.get_union.PreCodegen.after.mir
fn get_union() -> Foo {
    Foo { x: () }
}

fn main() {
    get_union();
}
