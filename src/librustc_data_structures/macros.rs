/// A simple static assertion macro.
#[macro_export]
#[allow_internal_unstable(type_ascription)]
macro_rules! static_assert {
    ($test:expr) => {
        // Use the bool to access an array such that if the bool is false, the access
        // is out-of-bounds.
        #[allow(dead_code)]
        const _: () = [()][!($test: bool) as usize];
    };
}

/// Type size assertion. The first argument is a type and the second argument is its expected size.
#[macro_export]
macro_rules! static_assert_size {
    ($ty:ty, $size:expr) => {
        const _: [(); $size] = [(); ::std::mem::size_of::<$ty>()];
    };
}

#[macro_export]
macro_rules! enum_from_u32 {
    ($(#[$attr:meta])* pub enum $name:ident {
        $($variant:ident = $e:expr,)*
    }) => {
        $(#[$attr])*
        pub enum $name {
            $($variant = $e),*
        }

        impl $name {
            pub fn from_u32(u: u32) -> Option<$name> {
                $(if u == $name::$variant as u32 {
                    return Some($name::$variant)
                })*
                None
            }
        }
    };
    ($(#[$attr:meta])* pub enum $name:ident {
        $($variant:ident,)*
    }) => {
        $(#[$attr])*
        pub enum $name {
            $($variant,)*
        }

        impl $name {
            pub fn from_u32(u: u32) -> Option<$name> {
                $(if u == $name::$variant as u32 {
                    return Some($name::$variant)
                })*
                None
            }
        }
    }
}
