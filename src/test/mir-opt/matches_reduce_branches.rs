// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR matches_reduce_branches.foo.MatchBranchSimplification.diff

fn foo(bar: Option<()>) {
    if matches!(bar, None) {
      ()
    }
}

fn main() {
  let _ = foo(None);
  let _ = foo(Some(()));
}
