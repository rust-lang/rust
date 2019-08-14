// run-pass
#![feature(transparent_unions)]

use std::mem::size_of;
use std::num::NonZeroUsize;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

trait Trait { fn dummy(&self) { } }
trait Mirror { type Image; }
impl<T> Mirror for T { type Image = T; }
struct ParamTypeStruct<T>(T);
struct AssocTypeStruct<T>(<T as Mirror>::Image);
#[repr(transparent)]
union MaybeUninitUnion<T: Copy> {
    _value: T,
    _uninit: (),
}

fn main() {
    // Functions
    assert_eq!(size_of::<fn(isize)>(), size_of::<Option<fn(isize)>>());
    assert_eq!(size_of::<extern "C" fn(isize)>(), size_of::<Option<extern "C" fn(isize)>>());

    // Slices - &str / &[T] / &mut [T]
    assert_eq!(size_of::<&str>(), size_of::<Option<&str>>());
    assert_eq!(size_of::<&[isize]>(), size_of::<Option<&[isize]>>());
    assert_eq!(size_of::<&mut [isize]>(), size_of::<Option<&mut [isize]>>());

    // Traits - Box<Trait> / &Trait / &mut Trait
    assert_eq!(size_of::<Box<dyn Trait>>(), size_of::<Option<Box<dyn Trait>>>());
    assert_eq!(size_of::<&dyn Trait>(), size_of::<Option<&dyn Trait>>());
    assert_eq!(size_of::<&mut dyn Trait>(), size_of::<Option<&mut dyn Trait>>());

    // Pointers - Box<T>
    assert_eq!(size_of::<Box<isize>>(), size_of::<Option<Box<isize>>>());

    // The optimization can't apply to raw pointers or unions with a ZST field.
    assert!(size_of::<Option<*const isize>>() != size_of::<*const isize>());
    assert!(Some(std::ptr::null::<isize>()).is_some()); // Can't collapse None to null
    assert_ne!(size_of::<fn(isize)>(), size_of::<Option<MaybeUninitUnion<fn(isize)>>>());
    assert_ne!(size_of::<&str>(), size_of::<Option<MaybeUninitUnion<&str>>>());
    assert_ne!(size_of::<NonNull<isize>>(), size_of::<Option<MaybeUninitUnion<NonNull<isize>>>>());

    struct Foo {
        _a: Box<isize>
    }
    struct Bar(Box<isize>);

    // Should apply through structs
    assert_eq!(size_of::<Foo>(), size_of::<Option<Foo>>());
    assert_eq!(size_of::<Bar>(), size_of::<Option<Bar>>());
    // and tuples
    assert_eq!(size_of::<(u8, Box<isize>)>(), size_of::<Option<(u8, Box<isize>)>>());
    // and fixed-size arrays
    assert_eq!(size_of::<[Box<isize>; 1]>(), size_of::<Option<[Box<isize>; 1]>>());

    // Should apply to NonZero
    assert_eq!(size_of::<NonZeroUsize>(), size_of::<Option<NonZeroUsize>>());
    assert_eq!(size_of::<NonNull<i8>>(), size_of::<Option<NonNull<i8>>>());

    // Should apply to types that use NonZero internally
    assert_eq!(size_of::<Vec<isize>>(), size_of::<Option<Vec<isize>>>());
    assert_eq!(size_of::<Arc<isize>>(), size_of::<Option<Arc<isize>>>());
    assert_eq!(size_of::<Rc<isize>>(), size_of::<Option<Rc<isize>>>());

    // Should apply to types that have NonZero transitively
    assert_eq!(size_of::<String>(), size_of::<Option<String>>());

    // Should apply to types where the pointer is substituted
    assert_eq!(size_of::<&u8>(), size_of::<Option<ParamTypeStruct<&u8>>>());
    assert_eq!(size_of::<&u8>(), size_of::<Option<AssocTypeStruct<&u8>>>());
}
