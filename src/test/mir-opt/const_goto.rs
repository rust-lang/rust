pub enum Foo {
    A,
    B,
    C,
    D,
    E,
    F,
}

// EMIT_MIR const_goto.issue_77355_opt.ConstGoto.diff
fn issue_77355_opt(num: Foo) -> u64 {
    if matches!(num, Foo::B | Foo::C) { 23 } else { 42 }
}
fn main() {
    issue_77355_opt(Foo::A);
}
