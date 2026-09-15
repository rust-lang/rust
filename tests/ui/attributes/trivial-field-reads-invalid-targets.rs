//! Checks that `#[rustc_trivial_field_reads]` are not allowed due to invalid targets

#![feature(rustc_attrs)]
#![deny(dead_code)]

trait Whatever {
    #[rustc_trivial_field_reads] //~ERROR the `rustc_trivial_field_reads` attribute cannot be used on required trait methods
    fn read(&self);
}

#[rustc_trivial_field_reads] //~ERROR the `rustc_trivial_field_reads` attribute cannot be used on traits
trait Whatever1 {}

#[rustc_trivial_field_reads] //~ERROR the `rustc_trivial_field_reads` attribute cannot be used on macro defs
  macro_rules! number {
      () => {
          67
      };
  }

fn main() {
    let _ = number!();
}
