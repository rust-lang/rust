//! Real world regressions and issues, not particularly minimized.
//!
//! While it's OK to just dump large macros here, it's preferable to come up
//! with a minimal example for the program and put a specific test to the parent
//! directory.

use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn test_vec() {
    check(
        r#"
macro_rules! vec {
   ($($item:expr),*) => {{
           let mut v = Vec::new();
           $( v.push($item); )*
           v
    }};
}
fn main() {
    vec!();
    vec![1u32,2];
}
"#,
        expect![[r#"
macro_rules! vec {
   ($($item:expr),*) => {{
           let mut v = Vec::new();
           $( v.push($item); )*
           v
    }};
}
fn main() {
     {
        let mut v = Vec::new();
        v
    };
     {
        let mut v = Vec::new();
        v.push(1u32);
        v.push(2);
        v
    };
}
"#]],
    );
}

#[test]
fn test_winapi_struct() {
    // from https://github.com/retep998/winapi-rs/blob/a7ef2bca086aae76cf6c4ce4c2552988ed9798ad/src/macros.rs#L366

    check(
        r#"
macro_rules! STRUCT {
    ($(#[$attrs:meta])* struct $name:ident {
        $($field:ident: $ftype:ty,)+
    }) => (
        #[repr(C)] #[derive(Copy)] $(#[$attrs])*
        pub struct $name {
            $(pub $field: $ftype,)+
        }
        impl Clone for $name {
            #[inline]
            fn clone(&self) -> $name { *self }
        }
        #[cfg(feature = "impl-default")]
        impl Default for $name {
            #[inline]
            fn default() -> $name { unsafe { $crate::_core::mem::zeroed() } }
        }
    );
}

// from https://github.com/retep998/winapi-rs/blob/a7ef2bca086aae76cf6c4ce4c2552988ed9798ad/src/shared/d3d9caps.rs
STRUCT!{struct D3DVSHADERCAPS2_0 {Caps: u8,}}

STRUCT!{#[cfg_attr(target_arch = "x86", repr(packed))] struct D3DCONTENTPROTECTIONCAPS {Caps : u8 ,}}
"#,
        expect![[r##"
macro_rules! STRUCT {
    ($(#[$attrs:meta])* struct $name:ident {
        $($field:ident: $ftype:ty,)+
    }) => (
        #[repr(C)] #[derive(Copy)] $(#[$attrs])*
        pub struct $name {
            $(pub $field: $ftype,)+
        }
        impl Clone for $name {
            #[inline]
            fn clone(&self) -> $name { *self }
        }
        #[cfg(feature = "impl-default")]
        impl Default for $name {
            #[inline]
            fn default() -> $name { unsafe { $crate::_core::mem::zeroed() } }
        }
    );
}

#[repr(C)]
#[derive(Copy)] pub struct D3DVSHADERCAPS2_0 {
    pub Caps: u8,
}
impl Clone for D3DVSHADERCAPS2_0 {
    #[inline] fn clone(&self ) -> D3DVSHADERCAPS2_0 {
        *self
    }
}
#[cfg(feature = "impl-default")] impl Default for D3DVSHADERCAPS2_0 {
    #[inline] fn default() -> D3DVSHADERCAPS2_0 {
        unsafe {
            $crate::_core::mem::zeroed()
        }
    }
}

#[repr(C)]
#[derive(Copy)]
#[cfg_attr(target_arch = "x86", repr(packed))] pub struct D3DCONTENTPROTECTIONCAPS {
    pub Caps: u8,
}
impl Clone for D3DCONTENTPROTECTIONCAPS {
    #[inline] fn clone(&self ) -> D3DCONTENTPROTECTIONCAPS {
        *self
    }
}
#[cfg(feature = "impl-default")] impl Default for D3DCONTENTPROTECTIONCAPS {
    #[inline] fn default() -> D3DCONTENTPROTECTIONCAPS {
        unsafe {
            $crate::_core::mem::zeroed()
        }
    }
}
"##]],
    );
}

#[test]
fn test_int_base() {
    check(
        r#"
macro_rules! int_base {
    ($Trait:ident for $T:ident as $U:ident -> $Radix:ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl fmt::$Trait for $T {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                $Radix.fmt_int(*self as $U, f)
            }
        }
    }
}
int_base!{Binary for isize as usize -> Binary}
"#,
        expect![[r##"
macro_rules! int_base {
    ($Trait:ident for $T:ident as $U:ident -> $Radix:ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl fmt::$Trait for $T {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                $Radix.fmt_int(*self as $U, f)
            }
        }
    }
}
#[stable(feature = "rust1", since = "1.0.0")] impl fmt::Binary for isize {
    fn fmt(&self , f: &mut fmt::Formatter< '_>) -> fmt::Result {
        Binary.fmt_int(*self as usize, f)
    }
}
"##]],
    );
}
