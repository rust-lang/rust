//@ run-pass
//@ ignore-backends: gcc
#![feature(c_variadic)]

#[repr(transparent)]
struct S(i32);

impl S {
    unsafe extern "C" fn associated_function(mut ap: ...) -> i32 {
        unsafe { ap.arg() }
    }

    unsafe extern "C" fn method_owned(self, mut ap: ...) -> i32 {
        self.0 + unsafe { ap.arg::<i32>() }
    }

    unsafe extern "C" fn method_ref(&self, mut ap: ...) -> i32 {
        self.0 + unsafe { ap.arg::<i32>() }
    }

    unsafe extern "C" fn method_mut(&mut self, mut ap: ...) -> i32 {
        self.0 + unsafe { ap.arg::<i32>() }
    }

    unsafe extern "C" fn fat_pointer(self: Box<Self>, mut ap: ...) -> i32 {
        self.0 + unsafe { ap.arg::<i32>() }
    }
}

fn main() {
    unsafe {
        assert_eq!(S::associated_function(32), 32);
        assert_eq!(S(100).method_owned(32), 132);
        assert_eq!(S(100).method_ref(32), 132);
        assert_eq!(S(100).method_mut(32), 132);
        assert_eq!(S::fat_pointer(Box::new(S(100)), 32), 132);

        type Method<T> = unsafe extern "C" fn(T, ...) -> i32;

        assert_eq!((S::associated_function as unsafe extern "C" fn(...) -> i32)(32), 32);
        assert_eq!((S::method_owned as Method<_>)(S(100), 32), 132);
        assert_eq!((S::method_ref as Method<_>)(&S(100), 32), 132);
        assert_eq!((S::method_mut as Method<_>)(&mut S(100), 32), 132);
        assert_eq!((S::fat_pointer as Method<_>)(Box::new(S(100)), 32), 132);
    }
}
