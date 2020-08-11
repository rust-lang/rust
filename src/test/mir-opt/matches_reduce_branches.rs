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
