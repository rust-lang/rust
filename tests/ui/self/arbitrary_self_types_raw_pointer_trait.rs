//@ run-pass
#![feature(arbitrary_self_types_pointers)]

use std::ptr;

trait Foo {
    fn foo(self: *const Self) -> &'static str;

    unsafe fn bar(self: *const Self) -> i64;

    unsafe fn complicated(self: *const *const Self) -> i64 where Self: Sized {
        (*self).bar()
    }
}

impl Foo for i32 {
    fn foo(self: *const Self) -> &'static str {
        "I'm an i32!"
    }

    unsafe fn bar(self: *const Self) -> i64 {
        *self as i64
    }
}

impl Foo for u32 {
    fn foo(self: *const Self) -> &'static str {
        "I'm a u32!"
    }

    unsafe fn bar(self: *const Self) -> i64 {
        *self as i64
    }
}

fn main() {
    let null_i32 = ptr::null::<i32>() as *const dyn Foo;
    let null_u32 = ptr::null::<u32>() as *const dyn Foo;

    assert_eq!("I'm an i32!", null_i32.foo());
    assert_eq!("I'm a u32!", null_u32.foo());

    let valid_i32 = 5i32;
    let valid_i32_thin = &valid_i32 as *const i32;
    assert_eq!("I'm an i32!", valid_i32_thin.foo());
    assert_eq!(5, unsafe { valid_i32_thin.bar() });
    assert_eq!(5, unsafe { (&valid_i32_thin as *const *const i32).complicated() });
    let valid_i32_fat = valid_i32_thin as *const dyn Foo;
    assert_eq!("I'm an i32!", valid_i32_fat.foo());
    assert_eq!(5, unsafe { valid_i32_fat.bar() });

    let valid_u32 = 18u32;
    let valid_u32_thin = &valid_u32 as *const u32;
    assert_eq!("I'm a u32!", valid_u32_thin.foo());
    assert_eq!(18, unsafe { valid_u32_thin.bar() });
    assert_eq!(18, unsafe { (&valid_u32_thin as *const *const u32).complicated() });
    let valid_u32_fat = valid_u32_thin as *const dyn Foo;
    assert_eq!("I'm a u32!", valid_u32_fat.foo());
    assert_eq!(18, unsafe { valid_u32_fat.bar() });

}
