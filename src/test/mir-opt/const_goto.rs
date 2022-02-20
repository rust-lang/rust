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

// EMIT_MIR const_goto.match_nested_if.MatchBranchSimplification.diff
fn match_nested_if() -> bool {
    let val = match () {
        () if if if if true { true } else { false } { true } else { false } {
            true
        } else {
            false
        } =>
        {
            true
        }
        _ => false,
    };
    val
}

fn main() {
    issue_77355_opt(Foo::A);
    let _ = match_nested_if();
}
