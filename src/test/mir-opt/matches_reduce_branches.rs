// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR matches_reduce_branches.foo.MatchBranchSimplification.diff
// EMIT_MIR matches_reduce_branches.bar.MatchBranchSimplification.diff
// EMIT_MIR matches_reduce_branches.match_nested_if.MatchBranchSimplification.diff

fn foo(bar: Option<()>) {
    if matches!(bar, None) {
      ()
    }
}

fn bar(i: i32) -> (bool, bool, bool, bool) {
    let a;
    let b;
    let c;
    let d;

    match i {
        7 => {
            a = false;
            b = true;
            c = false;
            d = true;
            ()
        }
        _ => {
            a = true;
            b = false;
            c = false;
            d = true;
            ()
        }
    };

    (a, b, c, d)
}

fn match_nested_if() -> bool {
    let val = match () {
        () if if if if true {true} else {false} {true} else {false} {true} else {false} => true,
        _ => false,
    };
    val
}

fn main() {
  let _ = foo(None);
  let _ = foo(Some(()));
  let _ = bar(0);
  let _ = match_nested_if();
}
