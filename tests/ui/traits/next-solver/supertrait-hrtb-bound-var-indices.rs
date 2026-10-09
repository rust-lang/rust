//@ check-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver=globally

trait Triple<'a, 'b, 'c> {}

trait Double<'a, 'b>: for<'c> Triple<'a, 'b, 'c> {}

fn needs_triple<T: for<'a, 'b, 'c> Triple<'a, 'b, 'c>>() {}

fn implied_supertrait<T: for<'a, 'b> Double<'a, 'b>>() {
    needs_triple::<T>();
}

fn main() {}
