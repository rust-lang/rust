//@ run-pass
// Test that type IDs correctly account for higher-rank lifetimes
// Also acts as a regression test for an ICE (issue #19791)

use std::any::{Any, TypeId};

struct Struct<'a>(#[allow(dead_code)] &'a ());
trait Trait<'a> {}

fn main() {
    // Bare fns
    {
        let a = TypeId::of::<fn(&'static isize, &'static isize)>();
        let b = TypeId::of::<for<'a> fn(&'static isize, &'a isize)>();
        let c = TypeId::of::<for<'a, 'b> fn(&'a isize, &'b isize)>();
        let d = TypeId::of::<for<'a, 'b> fn(&'b isize, &'a isize)>();
        assert!(a != b);
        assert!(a != c);
        assert!(a != d);
        assert!(b != c);
        assert!(b != d);
        assert_eq!(c, d);

        // Make sure De Bruijn indices are handled correctly
        let e = TypeId::of::<for<'a> fn(fn(&'a isize) -> &'a isize)>();
        let f = TypeId::of::<fn(for<'a> fn(&'a isize) -> &'a isize)>();
        assert!(e != f);

        // Make sure lifetime parameters of items are not ignored.
        let g = TypeId::of::<for<'a> fn(&'a dyn Trait<'a>) -> Struct<'a>>();
        let h = TypeId::of::<for<'a> fn(&'a dyn Trait<'a>) -> Struct<'static>>();
        let i = TypeId::of::<for<'a, 'b> fn(&'a dyn Trait<'b>) -> Struct<'b>>();
        assert!(g != h);
        assert!(g != i);
        assert!(h != i);

        // Make sure lifetime anonymization handles nesting correctly
        let j = TypeId::of::<fn(for<'a> fn(&'a isize) -> &'a usize)>();
        let k = TypeId::of::<fn(for<'b> fn(&'b isize) -> &'b usize)>();
        assert_eq!(j, k);
    }
    // Boxed unboxed closures
    {
        let a = TypeId::of::<Box<dyn Fn(&'static isize, &'static isize)>>();
        let b = TypeId::of::<Box<dyn for<'a> Fn(&'static isize, &'a isize)>>();
        let c = TypeId::of::<Box<dyn for<'a, 'b> Fn(&'a isize, &'b isize)>>();
        let d = TypeId::of::<Box<dyn for<'a, 'b> Fn(&'b isize, &'a isize)>>();
        assert!(a != b);
        assert!(a != c);
        assert!(a != d);
        assert!(b != c);
        assert!(b != d);
        assert_eq!(c, d);

        // Make sure De Bruijn indices are handled correctly
        let e = TypeId::of::<Box<dyn for<'a> Fn(Box<dyn Fn(&'a isize) -> &'a isize>)>>();
        let f = TypeId::of::<Box<dyn Fn(Box<dyn for<'a> Fn(&'a isize) -> &'a isize>)>>();
        assert!(e != f);
    }
    // Raw unboxed closures
    // Note that every unboxed closure has its own anonymous type,
    // so no two IDs should equal each other, even when compatible
    {
        let a = id(|_: &isize, _: &isize| {});
        let b = id(|_: &isize, _: &isize| {});
        assert!(a != b);
    }

    fn id<T:Any>(_: T) -> TypeId {
        TypeId::of::<T>()
    }
}
