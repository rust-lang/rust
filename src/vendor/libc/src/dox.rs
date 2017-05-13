pub use self::imp::*;

#[cfg(not(dox))]
mod imp {
    pub use core::option::Option;
    pub use core::clone::Clone;
    pub use core::marker::Copy;
    pub use core::mem;
}

#[cfg(dox)]
mod imp {
    pub enum Option<T> {
        Some(T),
        None,
    }
    impl<T: Copy> Copy for Option<T> {}
    impl<T: Clone> Clone for Option<T> {
        fn clone(&self) -> Option<T> { loop {} }
    }

    pub trait Clone {
        fn clone(&self) -> Self;
    }

    #[lang = "copy"]
    pub trait Copy {}

    #[lang = "sync"]
    pub trait Sync {}
    impl<T> Sync for T {}

    #[lang = "sized"]
    pub trait Sized {}

    macro_rules! each_int {
        ($mac:ident) => (
            $mac!(u8);
            $mac!(u16);
            $mac!(u32);
            $mac!(u64);
            $mac!(usize);
            $mac!(i8);
            $mac!(i16);
            $mac!(i32);
            $mac!(i64);
            $mac!(isize);
        )
    }

    #[lang = "div"]
    pub trait Div<RHS> {
        type Output;
        fn div(self, rhs: RHS) -> Self::Output;
    }

    macro_rules! impl_div {
        ($($i:ident)*) => ($(
            impl Div<$i> for $i {
                type Output = $i;
                fn div(self, rhs: $i) -> $i { self / rhs }
            }
        )*)
    }
    each_int!(impl_div);

    #[lang = "shl"]
    pub trait Shl<RHS> {
        type Output;
        fn shl(self, rhs: RHS) -> Self::Output;
    }

    macro_rules! impl_shl {
        ($($i:ident)*) => ($(
            impl Shl<$i> for $i {
                type Output = $i;
                fn shl(self, rhs: $i) -> $i { self << rhs }
            }
        )*)
    }
    each_int!(impl_shl);

    #[lang = "mul"]
    pub trait Mul<RHS=Self> {
        type Output;
        fn mul(self, rhs: RHS) -> Self::Output;
    }

    macro_rules! impl_mul {
        ($($i:ident)*) => ($(
            impl Mul for $i {
                type Output = $i;
                fn mul(self, rhs: $i) -> $i { self * rhs }
            }
        )*)
    }
    each_int!(impl_mul);

    #[lang = "sub"]
    pub trait Sub<RHS=Self> {
        type Output;
        fn sub(self, rhs: RHS) -> Self::Output;
    }

    macro_rules! impl_sub {
        ($($i:ident)*) => ($(
            impl Sub for $i {
                type Output = $i;
                fn sub(self, rhs: $i) -> $i { self - rhs }
            }
        )*)
    }
    each_int!(impl_sub);

    #[lang = "bitor"]
    pub trait Bitor<RHS=Self> {
        type Output;
        fn bitor(self, rhs: RHS) -> Self::Output;
    }

    macro_rules! impl_bitor {
        ($($i:ident)*) => ($(
            impl Bitor for $i {
                type Output = $i;
                fn bitor(self, rhs: $i) -> $i { self | rhs }
            }
        )*)
    }
    each_int!(impl_bitor);

    pub mod mem {
        pub fn size_of_val<T>(_: &T) -> usize { 4 }
    }
}
