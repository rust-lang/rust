// EMIT_MIR issue_110508.{impl#0}-BAR1.built.after.mir
// EMIT_MIR issue_110508.{impl#0}-BAR2.built.after.mir

enum Foo {
    Bar(()),
}

impl Foo {
    const BAR1: Foo = Foo::Bar(());
    const BAR2: Foo = Self::Bar(());
}

fn main() {}
