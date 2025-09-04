//@ only-x86_64-unknown-linux-gnu
//@ compile-flags: -C panic=abort -Zinline-mir=no -Copt-level=0 -Zcross-crate-inline-threshold=never -Zmir-opt-level=0 -Cno-prepopulate-passes
//@ no-prefer-dynamic
//@ edition:2024
#![crate_type = "lib"]

trait TestTrait {
    fn test_func(&self);
}

struct TestStruct {}

impl TestTrait for TestStruct {
    fn test_func(&self) {
        println!("TestStruct::test_func");
    }
}

#[inline(never)]
pub fn foo() -> impl TestTrait {
    TestStruct {}
}

//~ MONO_ITEM fn foo
//~ MONO_ITEM fn <TestStruct as TestTrait>::test_func

trait TestTrait2 {
    fn test_func2(&self);
}

struct TestStruct2 {}

impl TestTrait2 for TestStruct2 {
    fn test_func2(&self) {
        println!("TestStruct2::test_func2");
    }
}

#[inline(never)]
pub fn foo2() -> Box<dyn TestTrait2> {
    Box::new(TestStruct2 {})
}

//~ MONO_ITEM fn <TestStruct2 as TestTrait2>::test_func2
//~ MONO_ITEM fn alloc::alloc::exchange_malloc
//~ MONO_ITEM fn foo2
//~ MONO_ITEM fn std::alloc::Global::alloc_impl
//~ MONO_ITEM fn std::boxed::Box::<TestStruct2>::new
//~ MONO_ITEM fn std::alloc::Layout::from_size_align_unchecked::precondition_check
//~ MONO_ITEM fn std::ptr::NonNull::<T>::new_unchecked::precondition_check

struct Counter {
    count: usize,
}

impl Counter {
    fn new() -> Counter {
        Counter { count: 0 }
    }
}

impl Iterator for Counter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        if self.count < 6 { Some(self.count) } else { None }
    }
}

#[inline(never)]
pub fn foo3() -> Box<dyn Iterator<Item = usize>> {
    Box::new(Counter::new())
}

//~ MONO_ITEM fn <Counter as std::iter::Iterator::advance_by::SpecAdvanceBy>::spec_advance_by
//~ MONO_ITEM fn <Counter as std::iter::Iterator::advance_by::SpecAdvanceBy>::spec_advance_by::{closure#0}
//~ MONO_ITEM fn <Counter as std::iter::Iterator>::advance_by
//~ MONO_ITEM fn <Counter as std::iter::Iterator>::next
//~ MONO_ITEM fn <Counter as std::iter::Iterator>::nth
//~ MONO_ITEM fn <Counter as std::iter::Iterator>::size_hint
//~ MONO_ITEM fn <Counter as std::iter::Iterator>::try_fold::<std::num::NonZero<usize>, {closure@<Counter as std::iter::Iterator::advance_by::SpecAdvanceBy>::spec_advance_by::{closure#0}}, std::option::Option<std::num::NonZero<usize>>>
//~ MONO_ITEM fn <std::option::Option<std::num::NonZero<usize>> as std::ops::FromResidual<std::option::Option<std::convert::Infallible>>>::from_residual
//~ MONO_ITEM fn <std::option::Option<std::num::NonZero<usize>> as std::ops::Try>::branch
//~ MONO_ITEM fn <std::option::Option<std::num::NonZero<usize>> as std::ops::Try>::from_output
//~ MONO_ITEM fn foo3
//~ MONO_ITEM fn std::boxed::Box::<Counter>::new
//~ MONO_ITEM fn Counter::new
//~ MONO_ITEM fn core::fmt::rt::<impl std::fmt::Arguments<'_>>::new_const::<1>
