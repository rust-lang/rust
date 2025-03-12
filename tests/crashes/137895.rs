//@ known-bug: #137895
#![feature(sized_hierarchy)]

trait A {
    fn b() -> impl std::marker::PointeeSized + 'a;
}

impl A for dyn A {}
