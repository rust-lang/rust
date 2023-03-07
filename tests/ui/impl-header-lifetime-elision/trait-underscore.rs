// Test that `impl MyTrait<'_> for &i32` is equivalent to `impl<'a,
// 'b> MyTrait<'a> for &'b i32`.
//
// run-pass

#![allow(warnings)]

trait MyTrait<'a> { }

// This is equivalent to `MyTrait<'a> for &'b i32`, which is proven by
// the code below.
impl MyTrait<'_> for &i32 {
}

// When called, T will be `&'x i32` for some `'x`, so since we can
// prove that `&'x i32: for<'a> MyTrait<'a>, then we know that the
// lifetime parameter above is disconnected.
fn impls_my_trait<T: for<'a> MyTrait<'a>>() { }

fn impls_my_trait_val<T: for<'a> MyTrait<'a>>(_: T) {
    impls_my_trait::<T>();
}

fn random_where_clause()
where for<'a, 'b> &'a i32: MyTrait<'b> { }

fn main() {
    let x = 22;
    let f = &x;
    impls_my_trait_val(f);

    impls_my_trait::<&'static i32>();

    random_where_clause();
}
