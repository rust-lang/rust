//@ check-pass
#![crate_type = "lib"]
#![feature(sized_hierarchy)]

// Tests that a bound on an associated type projection, of a trait with a sizedness bound, will be
// elaborated.

trait FalseDeref {
    type Target: std::marker::PointeeSized;
}

trait Bar {}

fn foo<T: FalseDeref>()
where
    T::Target: Bar,
{
}
