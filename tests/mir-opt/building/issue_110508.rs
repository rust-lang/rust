// EMIT_MIR issue_110508.{impl#0}-BAR.built.after.mir
// EMIT_MIR issue_110508.{impl#0}-SELF_BAR.built.after.mir

enum Foo {
    Bar(()),
}

impl Foo {
    const BAR: Foo = Foo::Bar(());
    const SELF_BAR: Foo = Self::Bar(());
}

fn main() {}
