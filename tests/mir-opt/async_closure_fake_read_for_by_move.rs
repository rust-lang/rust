//@ edition:2021
// skip-filecheck

enum Foo {
    Bar,
    Baz,
}

// EMIT_MIR async_closure_fake_read_for_by_move.foo-{closure#0}-{closure#0}.built.after.mir
// EMIT_MIR async_closure_fake_read_for_by_move.foo-{closure#0}-{synthetic#0}.built.after.mir
fn foo(f: &Foo) {
    let x = async move || match f {
        Foo::Bar if true => {}
        _ => {}
    };
}

fn main() {}
