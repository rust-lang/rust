//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// cc #119820

trait Trait<'a, 'b> {}

trait OtherTrait<'b> {}
impl<'a, 'b, T: OtherTrait<'b>> Trait<'a, 'b> for T {}

fn impl_hr<'b, T: for<'a> Trait<'a, 'b>>() {}

fn not_hr<'a, T: for<'b> Trait<'a, 'b> + OtherTrait<'static>>() {
    impl_hr::<T>();
}

fn main() {}
