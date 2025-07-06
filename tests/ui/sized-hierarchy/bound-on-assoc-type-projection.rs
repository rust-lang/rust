//@ check-pass
#![crate_type = "lib"]
#![feature(sized_hierarchy)]

trait FalseDeref {
    type Target: std::marker::PointeeSized;
}

trait Bar {}

fn foo<T: FalseDeref>()
where
    T::Target: Bar,
{
}
