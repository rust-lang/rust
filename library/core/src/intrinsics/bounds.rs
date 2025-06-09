//! Various traits used to restrict intrinsics to not-completely-wrong types.

/// Types with a built-in dereference operator in runtime MIR,
/// aka references and raw pointers.
///
/// # Safety
/// Must actually *be* such a type.
pub unsafe trait BuiltinDeref: Sized {
    type Pointee: ?Sized;
}

unsafe impl<T: ?Sized> BuiltinDeref for &mut T {
    type Pointee = T;
}
unsafe impl<T: ?Sized> BuiltinDeref for &T {
    type Pointee = T;
}
unsafe impl<T: ?Sized> BuiltinDeref for *mut T {
    type Pointee = T;
}
unsafe impl<T: ?Sized> BuiltinDeref for *const T {
    type Pointee = T;
}

pub trait ChangePointee<U: ?Sized>: BuiltinDeref {
    type Output;
}
impl<'a, T: ?Sized + 'a, U: ?Sized + 'a> ChangePointee<U> for &'a mut T {
    type Output = &'a mut U;
}
impl<'a, T: ?Sized + 'a, U: ?Sized + 'a> ChangePointee<U> for &'a T {
    type Output = &'a U;
}
impl<T: ?Sized, U: ?Sized> ChangePointee<U> for *mut T {
    type Output = *mut U;
}
impl<T: ?Sized, U: ?Sized> ChangePointee<U> for *const T {
    type Output = *const U;
}
