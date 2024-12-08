// Test how overloaded deref interacts with borrows when DerefMut
// is implemented.

use std::ops::{Deref, DerefMut};

struct Own<T> {
    value: *mut T
}

impl<T> Deref for Own<T> {
    type Target = T;

    fn deref<'a>(&'a self) -> &'a T {
        unsafe { &*self.value }
    }
}

impl<T> DerefMut for Own<T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe { &mut *self.value }
    }
}

fn deref_imm(x: Own<isize>) {
    let __isize = &*x;
}

fn deref_mut1(x: Own<isize>) {
    let __isize = &mut *x; //~ ERROR cannot borrow
}

fn deref_mut2(mut x: Own<isize>) {
    let __isize = &mut *x;
}

fn deref_extend<'a>(x: &'a Own<isize>) -> &'a isize {
    &**x
}

fn deref_extend_mut1<'a>(x: &'a Own<isize>) -> &'a mut isize {
    &mut **x //~ ERROR cannot borrow
}

fn deref_extend_mut2<'a>(x: &'a mut Own<isize>) -> &'a mut isize {
    &mut **x
}

fn assign1<'a>(x: Own<isize>) {
    *x = 3; //~ ERROR cannot borrow
}

fn assign2<'a>(x: &'a Own<isize>) {
    **x = 3; //~ ERROR cannot borrow
}

fn assign3<'a>(x: &'a mut Own<isize>) {
    **x = 3;
}

pub fn main() {}
