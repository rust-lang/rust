pub struct Hidden;

#[macro_export]
macro_rules! definition_side {
    ($name:ident) => {
        pub fn $name() -> $crate::Hidden {
            loop {}
        }
    };
}

#[macro_export]
macro_rules! captured_type {
    ($name:ident, $ty:path) => {
        pub fn $name() -> $ty {
            loop {}
        }
    };
}

#[macro_export]
macro_rules! call_site_core {
    ($name:ident) => {
        pub fn $name() -> core::marker::PhantomData<()> {
            loop {}
        }
    };
}

#[macro_export]
macro_rules! nested {
    ($name:ident) => {
        $crate::definition_side!($name);
    };
}
