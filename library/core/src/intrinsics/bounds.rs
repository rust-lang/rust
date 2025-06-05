//! Various traits used to restrict intrinsics to not-completely-wrong types.

use crate::marker::PointeeSized;

/// Types with a built-in dereference operator in runtime MIR,
/// aka references and raw pointers.
///
/// # Safety
/// Must actually *be* such a type.
pub unsafe trait BuiltinDeref: Sized {
    type Pointee: PointeeSized;
}

unsafe impl<T: PointeeSized> BuiltinDeref for &mut T {
    type Pointee = T;
}
unsafe impl<T: PointeeSized> BuiltinDeref for &T {
    type Pointee = T;
}
unsafe impl<T: PointeeSized> BuiltinDeref for *mut T {
    type Pointee = T;
}
unsafe impl<T: PointeeSized> BuiltinDeref for *const T {
    type Pointee = T;
}

pub trait ChangePointee<U: PointeeSized>: BuiltinDeref {
    type Output;
}
impl<'a, T: PointeeSized + 'a, U: PointeeSized + 'a> ChangePointee<U> for &'a mut T {
    type Output = &'a mut U;
}
impl<'a, T: PointeeSized + 'a, U: PointeeSized + 'a> ChangePointee<U> for &'a T {
    type Output = &'a U;
}
impl<T: PointeeSized, U: PointeeSized> ChangePointee<U> for *mut T {
    type Output = *mut U;
}
impl<T: PointeeSized, U: PointeeSized> ChangePointee<U> for *const T {
    type Output = *const U;
}
