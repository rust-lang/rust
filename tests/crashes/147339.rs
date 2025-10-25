//@ known-bug: rust-lang/rust#147339
//@ compile-flags: --crate-type lib -Cinstrument-coverage
//@ needs-rustc-debug-assertions
macro_rules !foo {
  ($($m : ident $($f : ident $v : tt) +) *) => {
    $($(macro_rules !f{() =>{$v}}) +
      macro_rules !$m{() =>{$(fn f()->i32{$v}) + }}) *
  }
}
foo !(n c 3);
n!();
