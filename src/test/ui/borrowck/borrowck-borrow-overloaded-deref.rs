// Test how overloaded deref interacts with borrows when only
// Deref and not DerefMut is implemented.

use std::ops::Deref;

struct Rc<T> {
    value: *const T
}

impl<T> Deref for Rc<T> {
    type Target = T;

    fn deref<'a>(&'a self) -> &'a T {
        unsafe { &*self.value }
    }
}

fn deref_imm(x: Rc<isize>) {
    let __isize = &*x;
}

fn deref_mut1(x: Rc<isize>) {
    let __isize = &mut *x; //~ ERROR cannot borrow
}

fn deref_mut2(mut x: Rc<isize>) {
    let __isize = &mut *x; //~ ERROR cannot borrow
}

fn deref_extend<'a>(x: &'a Rc<isize>) -> &'a isize {
    &**x
}

fn deref_extend_mut1<'a>(x: &'a Rc<isize>) -> &'a mut isize {
    &mut **x //~ ERROR cannot borrow
}

fn deref_extend_mut2<'a>(x: &'a mut Rc<isize>) -> &'a mut isize {
    &mut **x //~ ERROR cannot borrow
}

fn assign1<'a>(x: Rc<isize>) {
    *x = 3; //~ ERROR cannot assign
}

fn assign2<'a>(x: &'a Rc<isize>) {
    **x = 3; //~ ERROR cannot assign
}

fn assign3<'a>(x: &'a mut Rc<isize>) {
    **x = 3; //~ ERROR cannot assign
}

pub fn main() {}
