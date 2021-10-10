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

#[test]
fn test_generate_pattern_iterators() {
    // From <https://github.com/rust-lang/rust/blob/316a391dcb7d66dc25f1f9a4ec9d368ef7615005/src/libcore/str/mod.rs>.
    check(
        r#"
macro_rules! generate_pattern_iterators {
    { double ended; with $(#[$common_stability_attribute:meta])*,
                        $forward_iterator:ident,
                        $reverse_iterator:ident, $iterty:ty
    } => { ok!(); }
}
generate_pattern_iterators ! ( double ended ; with # [ stable ( feature = "rust1" , since = "1.0.0" ) ] , Split , RSplit , & 'a str );
"#,
        expect![[r##"
macro_rules! generate_pattern_iterators {
    { double ended; with $(#[$common_stability_attribute:meta])*,
                        $forward_iterator:ident,
                        $reverse_iterator:ident, $iterty:ty
    } => { ok!(); }
}
ok!();
"##]],
    );
}

#[test]
fn test_impl_fn_for_zst() {
    // From <https://github.com/rust-lang/rust/blob/5d20ff4d2718c820632b38c1e49d4de648a9810b/src/libcore/internal_macros.rs>.
    check(
        r#"
macro_rules! impl_fn_for_zst  {
    {$( $( #[$attr: meta] )*
    struct $Name: ident impl$( <$( $lifetime : lifetime ),+> )? Fn =
        |$( $arg: ident: $ArgTy: ty ),*| -> $ReturnTy: ty $body: block;
    )+} => {$(
        $( #[$attr] )*
        struct $Name;

        impl $( <$( $lifetime ),+> )? Fn<($( $ArgTy, )*)> for $Name {
            #[inline]
            extern "rust-call" fn call(&self, ($( $arg, )*): ($( $ArgTy, )*)) -> $ReturnTy {
                $body
            }
        }

        impl $( <$( $lifetime ),+> )? FnMut<($( $ArgTy, )*)> for $Name {
            #[inline]
            extern "rust-call" fn call_mut(
                &mut self,
                ($( $arg, )*): ($( $ArgTy, )*)
            ) -> $ReturnTy {
                Fn::call(&*self, ($( $arg, )*))
            }
        }

        impl $( <$( $lifetime ),+> )? FnOnce<($( $ArgTy, )*)> for $Name {
            type Output = $ReturnTy;

            #[inline]
            extern "rust-call" fn call_once(self, ($( $arg, )*): ($( $ArgTy, )*)) -> $ReturnTy {
                Fn::call(&self, ($( $arg, )*))
            }
        }
    )+}
}

impl_fn_for_zst !   {
    #[derive(Clone)]
    struct CharEscapeDebugContinue impl Fn = |c: char| -> char::EscapeDebug {
        c.escape_debug_ext(false)
    };

    #[derive(Clone)]
    struct CharEscapeUnicode impl Fn = |c: char| -> char::EscapeUnicode {
        c.escape_unicode()
    };

    #[derive(Clone)]
    struct CharEscapeDefault impl Fn = |c: char| -> char::EscapeDefault {
        c.escape_default()
    };
}

"#,
        expect![[r##"
macro_rules! impl_fn_for_zst  {
    {$( $( #[$attr: meta] )*
    struct $Name: ident impl$( <$( $lifetime : lifetime ),+> )? Fn =
        |$( $arg: ident: $ArgTy: ty ),*| -> $ReturnTy: ty $body: block;
    )+} => {$(
        $( #[$attr] )*
        struct $Name;

        impl $( <$( $lifetime ),+> )? Fn<($( $ArgTy, )*)> for $Name {
            #[inline]
            extern "rust-call" fn call(&self, ($( $arg, )*): ($( $ArgTy, )*)) -> $ReturnTy {
                $body
            }
        }

        impl $( <$( $lifetime ),+> )? FnMut<($( $ArgTy, )*)> for $Name {
            #[inline]
            extern "rust-call" fn call_mut(
                &mut self,
                ($( $arg, )*): ($( $ArgTy, )*)
            ) -> $ReturnTy {
                Fn::call(&*self, ($( $arg, )*))
            }
        }

        impl $( <$( $lifetime ),+> )? FnOnce<($( $ArgTy, )*)> for $Name {
            type Output = $ReturnTy;

            #[inline]
            extern "rust-call" fn call_once(self, ($( $arg, )*): ($( $ArgTy, )*)) -> $ReturnTy {
                Fn::call(&self, ($( $arg, )*))
            }
        }
    )+}
}

#[derive(Clone)] struct CharEscapeDebugContinue;
impl Fn<(char, )> for CharEscapeDebugContinue {
    #[inline] extern "rust-call"fn call(&self , (c, ): (char, )) -> char::EscapeDebug { {
            c.escape_debug_ext(false )
        }
    }
}
impl FnMut<(char, )> for CharEscapeDebugContinue {
    #[inline] extern "rust-call"fn call_mut(&mut self , (c, ): (char, )) -> char::EscapeDebug {
        Fn::call(&*self , (c, ))
    }
}
impl FnOnce<(char, )> for CharEscapeDebugContinue {
    type Output = char::EscapeDebug;
    #[inline] extern "rust-call"fn call_once(self , (c, ): (char, )) -> char::EscapeDebug {
        Fn::call(&self , (c, ))
    }
}
#[derive(Clone)] struct CharEscapeUnicode;
impl Fn<(char, )> for CharEscapeUnicode {
    #[inline] extern "rust-call"fn call(&self , (c, ): (char, )) -> char::EscapeUnicode { {
            c.escape_unicode()
        }
    }
}
impl FnMut<(char, )> for CharEscapeUnicode {
    #[inline] extern "rust-call"fn call_mut(&mut self , (c, ): (char, )) -> char::EscapeUnicode {
        Fn::call(&*self , (c, ))
    }
}
impl FnOnce<(char, )> for CharEscapeUnicode {
    type Output = char::EscapeUnicode;
    #[inline] extern "rust-call"fn call_once(self , (c, ): (char, )) -> char::EscapeUnicode {
        Fn::call(&self , (c, ))
    }
}
#[derive(Clone)] struct CharEscapeDefault;
impl Fn<(char, )> for CharEscapeDefault {
    #[inline] extern "rust-call"fn call(&self , (c, ): (char, )) -> char::EscapeDefault { {
            c.escape_default()
        }
    }
}
impl FnMut<(char, )> for CharEscapeDefault {
    #[inline] extern "rust-call"fn call_mut(&mut self , (c, ): (char, )) -> char::EscapeDefault {
        Fn::call(&*self , (c, ))
    }
}
impl FnOnce<(char, )> for CharEscapeDefault {
    type Output = char::EscapeDefault;
    #[inline] extern "rust-call"fn call_once(self , (c, ): (char, )) -> char::EscapeDefault {
        Fn::call(&self , (c, ))
    }
}

"##]],
    );
}

#[test]
fn test_impl_nonzero_fmt() {
    // From <https://github.com/rust-lang/rust/blob/316a391dcb7d66dc25f1f9a4ec9d368ef7615005/src/libcore/num/mod.rs#L12>.
    check(
        r#"
macro_rules! impl_nonzero_fmt {
    ( #[$stability: meta] ( $( $Trait: ident ),+ ) for $Ty: ident ) => { ok!(); }
}
impl_nonzero_fmt! {
    #[stable(feature= "nonzero",since="1.28.0")]
    (Debug, Display, Binary, Octal, LowerHex, UpperHex) for NonZeroU8
}
"#,
        expect![[r##"
macro_rules! impl_nonzero_fmt {
    ( #[$stability: meta] ( $( $Trait: ident ),+ ) for $Ty: ident ) => { ok!(); }
}
ok!();
"##]],
    );
}
