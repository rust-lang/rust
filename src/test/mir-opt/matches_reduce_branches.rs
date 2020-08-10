// compile-flags: --emit mir
// EMIT_MIR matches_reduce_branches.foo.fix_match_arms.diff

fn foo(bar: Option<()>) {
    if matches!(bar, None) {
      ()
    }
}

fn main() {
  let _ = foo(None);
  let _ = foo(Some(()));
}
