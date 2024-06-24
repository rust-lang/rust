//! **FAKE** lazy_static crate.

#[macro_export]
macro_rules! lazy_static {
    (static ref $N:ident : $T:ty = $e:expr; $($t:tt)*) => {
        static $N : &::core::marker::PhantomData<$T> = &::core::marker::PhantomData;

        $crate::lazy_static! { $($t)* }
    };
    () => ()
}

#[macro_export]
macro_rules! external {
    () => {
        $crate::lazy_static! {
            static ref LZ_DERP: u32 = 12;
        }
    };
}
