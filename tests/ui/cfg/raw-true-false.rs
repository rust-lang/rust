//@ check-pass
//@ compile-flags: --cfg false --check-cfg=cfg(r#false)

#![deny(warnings)]

#[expect(unexpected_cfgs)]
mod a {
  #[cfg(r#true)]
  pub fn foo() {}
}

mod b {
  #[cfg(r#false)]
  pub fn bar() {}
}

fn main() {
    b::bar()
}
