//@ run-pass
//@ ignore-backends: gcc
#![feature(c_variadic)]

#[repr(transparent)]
struct Struct(i32);

impl Struct {
    unsafe extern "C" fn associated_function(mut ap: ...) -> i32 {
        unsafe { ap.arg() }
    }

    unsafe extern "C" fn method(&self, mut ap: ...) -> i32 {
        self.0 + unsafe { ap.arg::<i32>() }
    }
}

trait Trait: Sized {
    fn get(&self) -> i32;

    unsafe extern "C" fn trait_associated_function(mut ap: ...) -> i32 {
        unsafe { ap.arg() }
    }

    unsafe extern "C" fn trait_method_owned(self, mut ap: ...) -> i32 {
        self.get() + unsafe { ap.arg::<i32>() }
    }

    unsafe extern "C" fn trait_method_ref(&self, mut ap: ...) -> i32 {
        self.get() + unsafe { ap.arg::<i32>() }
    }

    unsafe extern "C" fn trait_method_mut(&mut self, mut ap: ...) -> i32 {
        self.get() + unsafe { ap.arg::<i32>() }
    }

    unsafe extern "C" fn trait_fat_pointer(self: Box<Self>, mut ap: ...) -> i32 {
        self.get() + unsafe { ap.arg::<i32>() }
    }
}

impl Trait for Struct {
    fn get(&self) -> i32 {
        self.0
    }
}

fn main() {
    unsafe {
        assert_eq!(Struct::associated_function(32), 32);
        assert_eq!(Struct(100).method(32), 132);

        assert_eq!(Struct::trait_associated_function(32), 32);
        assert_eq!(Struct(100).trait_method_owned(32), 132);
        assert_eq!(Struct(100).trait_method_ref(32), 132);
        assert_eq!(Struct(100).trait_method_mut(32), 132);
        assert_eq!(Struct::trait_fat_pointer(Box::new(Struct(100)), 32), 132);

        assert_eq!(<Struct as Trait>::trait_associated_function(32), 32);
        assert_eq!(Trait::trait_method_owned(Struct(100), 32), 132);
        assert_eq!(Trait::trait_method_ref(&Struct(100), 32), 132);
        assert_eq!(Trait::trait_method_mut(&mut Struct(100), 32), 132);
        assert_eq!(Trait::trait_fat_pointer(Box::new(Struct(100)), 32), 132);

        type Associated = unsafe extern "C" fn(...) -> i32;
        type Method<T> = unsafe extern "C" fn(T, ...) -> i32;

        assert_eq!((Struct::trait_associated_function as Associated)(32), 32);
        assert_eq!((Struct::trait_method_owned as Method<_>)(Struct(100), 32), 132);
        assert_eq!((Struct::trait_method_ref as Method<_>)(&Struct(100), 32), 132);
        assert_eq!((Struct::trait_method_mut as Method<_>)(&mut Struct(100), 32), 132);
        assert_eq!((Struct::trait_fat_pointer as Method<_>)(Box::new(Struct(100)), 32), 132);
    }
}
