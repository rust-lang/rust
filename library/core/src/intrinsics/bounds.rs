//! Various traits used to restrict intrinsics to not-completely-wrong types.

use crate::marker::{Forget, PointeeSized};

/// Types with a built-in dereference operator in runtime MIR,
/// aka references and raw pointers.
///
/// # Safety
/// Must actually *be* such a type.
pub unsafe trait BuiltinDeref: Sized + ?Forget {
    type Pointee: PointeeSized + ?Forget;
}

unsafe impl<T: PointeeSized + ?Forget> BuiltinDeref for &mut T {
    type Pointee = T;
}
unsafe impl<T: PointeeSized + ?Forget> BuiltinDeref for &T {
    type Pointee = T;
}
unsafe impl<T: PointeeSized + ?Forget> BuiltinDeref for *mut T {
    type Pointee = T;
}
unsafe impl<T: PointeeSized + ?Forget> BuiltinDeref for *const T {
    type Pointee = T;
}

pub trait ChangePointee<U: PointeeSized + ?Forget>: BuiltinDeref + ?Forget {
    type Output: ?Forget;
}
impl<'a, T: PointeeSized + ?Forget + 'a, U: PointeeSized + ?Forget + 'a> ChangePointee<U> for &'a mut T {
    type Output = &'a mut U;
}
impl<'a, T: PointeeSized + ?Forget + 'a, U: PointeeSized + ?Forget + 'a> ChangePointee<U> for &'a T {
    type Output = &'a U;
}
impl<T: PointeeSized + ?Forget, U: PointeeSized + ?Forget> ChangePointee<U> for *mut T {
    type Output = *mut U;
}
impl<T: PointeeSized + ?Forget, U: PointeeSized + ?Forget> ChangePointee<U> for *const T {
    type Output = *const U;
}
