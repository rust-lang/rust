#![warn(clippy::transmute_ptr_to_ref)]

unsafe fn _ptr_to_ref<T, U>(p: *const T, m: *mut T, o: *const U, om: *mut U) {
    let _: &T = std::mem::transmute(p);
    let _: &T = &*p;

    let _: &mut T = std::mem::transmute(m);
    let _: &mut T = &mut *m;

    let _: &T = std::mem::transmute(m);
    let _: &T = &*m;

    let _: &mut T = std::mem::transmute(p as *mut T);
    let _ = &mut *(p as *mut T);

    let _: &T = std::mem::transmute(o);
    let _: &T = &*(o as *const T);

    let _: &mut T = std::mem::transmute(om);
    let _: &mut T = &mut *(om as *mut T);

    let _: &T = std::mem::transmute(om);
    let _: &T = &*(om as *const T);
}

fn issue1231() {
    struct Foo<'a, T> {
        bar: &'a T,
    }

    let raw = 42 as *const i32;
    let _: &Foo<u8> = unsafe { std::mem::transmute::<_, &Foo<_>>(raw) };

    let _: &Foo<&u8> = unsafe { std::mem::transmute::<_, &Foo<&_>>(raw) };

    type Bar<'a> = &'a u8;
    let raw = 42 as *const i32;
    unsafe { std::mem::transmute::<_, Bar>(raw) };
}

fn main() {}
