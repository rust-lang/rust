//@ run-pass
#![feature(try_as_dyn)]

use std::fmt::Debug;

// Generic over `?Sized` T: relies on the relaxed bound on `try_as_dyn`.
fn try_debug<T: ?Sized + 'static>(t: &T) -> Option<String> {
    std::any::try_as_dyn::<T, dyn Debug>(t).map(|d| format!("{d:?}"))
}

fn try_debug_mut<T: ?Sized + 'static>(t: &mut T) -> Option<String> {
    std::any::try_as_dyn_mut::<T, dyn Debug>(t).map(|d| format!("{d:?}"))
}

fn main() {
    // Sized case still works through a `?Sized` generic context.
    let x: i32 = 7;
    assert_eq!(try_debug(&x).as_deref(), Some("7"));

    let mut y: i32 = 8;
    assert_eq!(try_debug_mut(&mut y).as_deref(), Some("8"));

    // Unsized `T` always returns `None`, even though `str: Debug` and
    // `[T]: Debug` hold — vtable lookup for unsized impl types is not
    // currently supported by `TypeId::trait_info_of`.
    let s: &str = "hello";
    assert!(try_debug::<str>(s).is_none());

    let slice: &[i32] = &[1, 2, 3];
    assert!(try_debug::<[i32]>(slice).is_none());

    let dyn_any: &dyn std::any::Any = &0i32;
    assert!(try_debug::<dyn std::any::Any>(dyn_any).is_none());
}
