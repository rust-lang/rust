// Tests to make sure that we reject polymorphic fn ptrs.

#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

fn foo() -> for<T> fn(T) {
    //~^ ERROR late-bound type parameter not allowed on function pointer types
    todo!()
}

fn main() {
    foo()(1i32);
}
