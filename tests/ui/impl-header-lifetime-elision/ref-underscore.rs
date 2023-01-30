// Test that `impl MyTrait for &i32` works and is equivalent to any lifetime.

// run-pass

#![allow(warnings)]

trait MyTrait { }

impl MyTrait for &i32 {
}

fn impls_my_trait<T: MyTrait>() { }

fn impls_my_trait_val<T: MyTrait>(_: T) {
    impls_my_trait::<T>();
}

fn random_where_clause()
where for<'a> &'a i32: MyTrait { }

fn main() {
    let x = 22;
    let f = &x;

    impls_my_trait_val(f);

    impls_my_trait::<&'static i32>();

    random_where_clause();
}
