//@ compile-flags: --diagnostic-width=60 -Zwrite-long-types-to-disk=yes
type A = (i32, i32, i32, i32);
type B = (A, A, A, A);
type C = (B, B, B, B);
type D = (C, C, C, C);

trait Trait {}

fn require_trait<T: Trait>() {}

fn main() {
    require_trait::<D>(); //~ ERROR the trait bound `(...
}
