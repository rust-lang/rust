#![warn(clippy::nonnull_unchecked_on_box_ptr)]

use std::ptr::NonNull;

macro_rules! identity {
    ($x:expr) => {
        $x
    };
}

macro_rules! weird {
    ($x:expr) => {{
        let y = 1;
        $x
    }};
}

macro_rules! from_macro {
    ($x:expr) => {
        unsafe { NonNull::new_unchecked(Box::into_raw($x)) }
    };
}

fn identity<T>(x: T) -> T {
    x
}

unsafe fn unsafe_identity<T>(x: T) -> T {
    x
}

fn lint() {
    fn basic() {
        let one = Box::new(1);
        let _ = unsafe {
            //~^ nonnull_unchecked_on_box_ptr
            NonNull::new_unchecked(Box::into_raw(one))
        };

        let one = Box::new(1);
        let _ = unsafe {
            //~^ nonnull_unchecked_on_box_ptr
            NonNull::new_unchecked(Box::into_raw(identity(one)))
        };
    }

    fn qualifiers() {
        let one = Box::new(1);
        let _ = unsafe {
            //~^ nonnull_unchecked_on_box_ptr
            std::ptr::NonNull::new_unchecked(std::boxed::Box::into_raw(one))
        };

        {
            use Box as Box2;
            use NonNull as NonNull2;
            let one = Box::new(1);
            let _ = unsafe {
                //~^ nonnull_unchecked_on_box_ptr
                NonNull2::new_unchecked(Box2::into_raw(one))
            };
        }

        {
            type Box2<T> = Box<T>;
            type NonNull2<T> = NonNull<T>;
            let one = Box::new(1);
            let _ = unsafe {
                //~^ nonnull_unchecked_on_box_ptr
                NonNull2::new_unchecked(Box2::into_raw(one))
            };
        }
    }

    fn macros() {
        let one = Box::new(1);
        let _ = unsafe {
            //~^ nonnull_unchecked_on_box_ptr
            NonNull::new_unchecked(Box::into_raw(identity!(one)))
        };

        let one = Box::new(1);
        let _ = identity!(unsafe {
            //~^ nonnull_unchecked_on_box_ptr
            NonNull::new_unchecked(Box::into_raw(one))
        });

        let one = Box::new(1);
        let _ = unsafe {
            //~^ nonnull_unchecked_on_box_ptr
            identity!(NonNull::new_unchecked(Box::into_raw(one)))
        };

        let one = Box::new(1);
        let _ = unsafe {
            //~^ nonnull_unchecked_on_box_ptr
            NonNull::new_unchecked(identity!(Box::into_raw(one)))
        };

        let one = Box::new(1);
        let _ = unsafe {
            //~^ nonnull_unchecked_on_box_ptr
            NonNull::new_unchecked(identity!(Box::into_raw(identity!(one))))
        };

        let one = Box::new(1);
        let _ = unsafe {
            //~^ nonnull_unchecked_on_box_ptr
            NonNull::new_unchecked(Box::into_raw(weird!(one)))
        };
    }

    fn keep_unsafe_block() {
        let one = Box::new(1);
        let _ = unsafe {
            NonNull::new_unchecked(Box::into_raw(unsafe_identity(one)))
            //~^ nonnull_unchecked_on_box_ptr
        };

        let one = Box::new(1);
        let _ = unsafe {
            identity(NonNull::new_unchecked(Box::into_raw(one)))
            //~^ nonnull_unchecked_on_box_ptr
        };

        let one = Box::new(1);
        let _ = unsafe {
            unsafe_identity(NonNull::new_unchecked(Box::into_raw(one)))
            //~^ nonnull_unchecked_on_box_ptr
        };

        let _ = unsafe {
            NonNull::new_unchecked(Box::into_raw(Box::new(std::num::NonZeroI32::new_unchecked(1))))
            //~^ nonnull_unchecked_on_box_ptr
        };

        let _ = unsafe {
            let one = Box::new(1);
            NonNull::new_unchecked(Box::into_raw(one))
            //~^ nonnull_unchecked_on_box_ptr
        };
    }
}

fn no_lint() {
    fn basic() {
        let one = Box::new(1);
        let _ = NonNull::from_mut(Box::leak(one));

        let one = Box::new(1);
        let _ = unsafe { NonNull::new_unchecked(identity(Box::into_raw(one))) };
    }

    // TODO?
    fn does_not_check_expr_init() {
        let one = Box::new(1);
        let leaked = Box::into_raw(one);
        let _ = unsafe { NonNull::new_unchecked(leaked) };

        let one = Box::new(1);
        let leaked = Box::leak(one);
        let _ = NonNull::from_mut(leaked);
    }

    fn macros() {
        let one = Box::new(1);
        let _ = from_macro!(one);
    }
}

#[clippy::msrv = "1.25"]
fn msrv_1_25() {
    let one = Box::new(1);
    let _ = unsafe { NonNull::new_unchecked(Box::into_raw(one)) };
}

#[clippy::msrv = "1.26"]
fn msrv_1_26() {
    let one = Box::new(1);
    let _ = unsafe {
        //~^ nonnull_unchecked_on_box_ptr
        NonNull::new_unchecked(Box::into_raw(one))
    };
}

#[clippy::msrv = "1.89"]
fn msrv_1_89() {
    let one = Box::new(1);
    let _ = unsafe {
        //~^ nonnull_unchecked_on_box_ptr
        NonNull::new_unchecked(Box::into_raw(one))
    };
}

fn main() {}
