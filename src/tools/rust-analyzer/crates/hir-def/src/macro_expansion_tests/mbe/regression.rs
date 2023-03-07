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
    fn fmt(&self , f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

#[test]
fn test_cfg_if_items() {
    // From <https://github.com/rust-lang/rust/blob/33fe1131cadba69d317156847be9a402b89f11bb/src/libstd/macros.rs#L986>.
    check(
        r#"
macro_rules! __cfg_if_items {
    (($($not:meta,)*) ; ) => {};
    (($($not:meta,)*) ; ( ($($m:meta),*) ($($it:item)*) ), $($rest:tt)*) => {
            __cfg_if_items! { ($($not,)* $($m,)*) ; $($rest)* }
    }
}
__cfg_if_items! {
    (rustdoc,);
    ( () (
           #[ cfg(any(target_os = "redox", unix))]
           #[ stable(feature = "rust1", since = "1.0.0")]
           pub use sys::ext as unix;

           #[cfg(windows)]
           #[stable(feature = "rust1", since = "1.0.0")]
           pub use sys::ext as windows;

           #[cfg(any(target_os = "linux", target_os = "l4re"))]
           pub mod linux;
    )),
}
"#,
        expect![[r#"
macro_rules! __cfg_if_items {
    (($($not:meta,)*) ; ) => {};
    (($($not:meta,)*) ; ( ($($m:meta),*) ($($it:item)*) ), $($rest:tt)*) => {
            __cfg_if_items! { ($($not,)* $($m,)*) ; $($rest)* }
    }
}
__cfg_if_items! {
    (rustdoc, );
}
"#]],
    );
}

#[test]
fn test_cfg_if_main() {
    // From <https://github.com/rust-lang/rust/blob/3d211248393686e0f73851fc7548f6605220fbe1/src/libpanic_unwind/macros.rs#L9>.
    check(
        r#"
macro_rules! cfg_if {
    ($(if #[cfg($($meta:meta),*)] { $($it:item)* } )else* else { $($it2:item)* })
    => {
        __cfg_if_items! {
            () ;
            $( ( ($($meta),*) ($($it)*) ), )*
            ( () ($($it2)*) ),
        }
    };

    // Internal macro to Apply a cfg attribute to a list of items
    (@__apply $m:meta, $($it:item)*) => { $(#[$m] $it)* };
}

cfg_if! {
    if #[cfg(target_env = "msvc")] {
        // no extra unwinder support needed
    } else if #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))] {
        // no unwinder on the system!
    } else {
        mod libunwind;
        pub use libunwind::*;
    }
}

cfg_if! {
    @__apply cfg(all(not(any(not(any(target_os = "solaris", target_os = "illumos")))))),
}
"#,
        expect![[r##"
macro_rules! cfg_if {
    ($(if #[cfg($($meta:meta),*)] { $($it:item)* } )else* else { $($it2:item)* })
    => {
        __cfg_if_items! {
            () ;
            $( ( ($($meta),*) ($($it)*) ), )*
            ( () ($($it2)*) ),
        }
    };

    // Internal macro to Apply a cfg attribute to a list of items
    (@__apply $m:meta, $($it:item)*) => { $(#[$m] $it)* };
}

__cfg_if_items! {
    ();
    ((target_env = "msvc")()), ((all(target_arch = "wasm32", not(target_os = "emscripten")))()), (()(mod libunwind;
    pub use libunwind::*;
    )),
}


"##]],
    );
}

#[test]
fn test_proptest_arbitrary() {
    // From <https://github.com/AltSysrq/proptest/blob/d1c4b049337d2f75dd6f49a095115f7c532e5129/proptest/src/arbitrary/macros.rs#L16>.
    check(
        r#"
macro_rules! arbitrary {
    ([$($bounds : tt)*] $typ: ty, $strat: ty, $params: ty;
        $args: ident => $logic: expr) => {
        impl<$($bounds)*> $crate::arbitrary::Arbitrary for $typ {
            type Parameters = $params;
            type Strategy = $strat;
            fn arbitrary_with($args: Self::Parameters) -> Self::Strategy {
                $logic
            }
        }
    };
}

arbitrary!(
    [A:Arbitrary]
    Vec<A> ,
    VecStrategy<A::Strategy>,
    RangedParams1<A::Parameters>;
    args =>   {
        let product_unpack![range, a] = args;
        vec(any_with::<A>(a), range)
    }
);
"#,
        expect![[r#"
macro_rules! arbitrary {
    ([$($bounds : tt)*] $typ: ty, $strat: ty, $params: ty;
        $args: ident => $logic: expr) => {
        impl<$($bounds)*> $crate::arbitrary::Arbitrary for $typ {
            type Parameters = $params;
            type Strategy = $strat;
            fn arbitrary_with($args: Self::Parameters) -> Self::Strategy {
                $logic
            }
        }
    };
}

impl <A: Arbitrary> $crate::arbitrary::Arbitrary for Vec<A> {
    type Parameters = RangedParams1<A::Parameters>;
    type Strategy = VecStrategy<A::Strategy>;
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy { {
            let product_unpack![range, a] = args;
            vec(any_with::<A>(a), range)
        }
    }
}
"#]],
    );
}

#[test]
fn test_old_ridl() {
    // This is from winapi 2.8, which do not have a link from github.
    check(
        r#"
#[macro_export]
macro_rules! RIDL {
    (interface $interface:ident ($vtbl:ident) : $pinterface:ident ($pvtbl:ident)
        {$(
            fn $method:ident(&mut self $(,$p:ident : $t:ty)*) -> $rtr:ty
        ),+}
    ) => {
        impl $interface {
            $(pub unsafe fn $method(&mut self) -> $rtr {
                ((*self.lpVtbl).$method)(self $(,$p)*)
            })+
        }
    };
}

RIDL!{interface ID3D11Asynchronous(ID3D11AsynchronousVtbl): ID3D11DeviceChild(ID3D11DeviceChildVtbl) {
    fn GetDataSize(&mut self) -> UINT
}}
"#,
        expect![[r##"
#[macro_export]
macro_rules! RIDL {
    (interface $interface:ident ($vtbl:ident) : $pinterface:ident ($pvtbl:ident)
        {$(
            fn $method:ident(&mut self $(,$p:ident : $t:ty)*) -> $rtr:ty
        ),+}
    ) => {
        impl $interface {
            $(pub unsafe fn $method(&mut self) -> $rtr {
                ((*self.lpVtbl).$method)(self $(,$p)*)
            })+
        }
    };
}

impl ID3D11Asynchronous {
    pub unsafe fn GetDataSize(&mut self ) -> UINT {
        ((*self .lpVtbl).GetDataSize)(self )
    }
}
"##]],
    );
}

#[test]
fn test_quick_error() {
    check(
        r#"
macro_rules! quick_error {
    (SORT [enum $name:ident $( #[$meta:meta] )*]
        items [$($( #[$imeta:meta] )*
                  => $iitem:ident: $imode:tt [$( $ivar:ident: $ityp:ty ),*]
                                {$( $ifuncs:tt )*} )* ]
        buf [ ]
        queue [ ]
    ) => {
        quick_error!(ENUMINITION [enum $name $( #[$meta] )*]
            body []
            queue [$(
                $( #[$imeta] )*
                =>
                $iitem: $imode [$( $ivar: $ityp ),*]
            )*]
        );
    };
}
quick_error ! (
    SORT
    [enum Wrapped #[derive(Debug)]]
    items [
        => One: UNIT [] {}
        => Two: TUPLE [s :String] {display ("two: {}" , s) from ()} ]
    buf [ ]
    queue [ ]
);

"#,
        expect![[r##"
macro_rules! quick_error {
    (SORT [enum $name:ident $( #[$meta:meta] )*]
        items [$($( #[$imeta:meta] )*
                  => $iitem:ident: $imode:tt [$( $ivar:ident: $ityp:ty ),*]
                                {$( $ifuncs:tt )*} )* ]
        buf [ ]
        queue [ ]
    ) => {
        quick_error!(ENUMINITION [enum $name $( #[$meta] )*]
            body []
            queue [$(
                $( #[$imeta] )*
                =>
                $iitem: $imode [$( $ivar: $ityp ),*]
            )*]
        );
    };
}
quick_error!(ENUMINITION[enum Wrapped#[derive(Debug)]]body[]queue[ = > One: UNIT[] = > Two: TUPLE[s: String]]);

"##]],
    )
}

#[test]
fn test_empty_repeat_vars_in_empty_repeat_vars() {
    check(
        r#"
macro_rules! delegate_impl {
    ([$self_type:ident, $self_wrap:ty, $self_map:ident]
     pub trait $name:ident $(: $sup:ident)* $(+ $more_sup:ident)* {

        $(
        @escape [type $assoc_name_ext:ident]
        )*
        $(
        @section type
        $(
            $(#[$_assoc_attr:meta])*
            type $assoc_name:ident $(: $assoc_bound:ty)*;
        )+
        )*
        $(
        @section self
        $(
            $(#[$_method_attr:meta])*
            fn $method_name:ident(self $(: $self_selftype:ty)* $(,$marg:ident : $marg_ty:ty)*) -> $mret:ty;
        )+
        )*
        $(
        @section nodelegate
        $($tail:tt)*
        )*
    }) => {
        impl<> $name for $self_wrap where $self_type: $name {
            $(
            $(
                fn $method_name(self $(: $self_selftype)* $(,$marg: $marg_ty)*) -> $mret {
                    $self_map!(self).$method_name($($marg),*)
                }
            )*
            )*
        }
    }
}
delegate_impl ! {
    [G, &'a mut G, deref] pub trait Data: GraphBase {@section type type NodeWeight;}
}
"#,
        expect![[r##"
macro_rules! delegate_impl {
    ([$self_type:ident, $self_wrap:ty, $self_map:ident]
     pub trait $name:ident $(: $sup:ident)* $(+ $more_sup:ident)* {

        $(
        @escape [type $assoc_name_ext:ident]
        )*
        $(
        @section type
        $(
            $(#[$_assoc_attr:meta])*
            type $assoc_name:ident $(: $assoc_bound:ty)*;
        )+
        )*
        $(
        @section self
        $(
            $(#[$_method_attr:meta])*
            fn $method_name:ident(self $(: $self_selftype:ty)* $(,$marg:ident : $marg_ty:ty)*) -> $mret:ty;
        )+
        )*
        $(
        @section nodelegate
        $($tail:tt)*
        )*
    }) => {
        impl<> $name for $self_wrap where $self_type: $name {
            $(
            $(
                fn $method_name(self $(: $self_selftype)* $(,$marg: $marg_ty)*) -> $mret {
                    $self_map!(self).$method_name($($marg),*)
                }
            )*
            )*
        }
    }
}
impl <> Data for &'amut G where G: Data {}
"##]],
    );
}

#[test]
fn test_issue_2520() {
    check(
        r#"
macro_rules! my_macro {
    {
        ( $(
            $( [] $sname:ident : $stype:ty  )?
            $( [$expr:expr] $nname:ident : $ntype:ty  )?
        ),* )
    } => {ok!(
        Test {
            $(
                $( $sname, )?
            )*
        }
    );};
}

my_macro! {
    ([] p1: u32, [|_| S0K0] s: S0K0, [] k0: i32)
}
    "#,
        expect![[r#"
macro_rules! my_macro {
    {
        ( $(
            $( [] $sname:ident : $stype:ty  )?
            $( [$expr:expr] $nname:ident : $ntype:ty  )?
        ),* )
    } => {ok!(
        Test {
            $(
                $( $sname, )?
            )*
        }
    );};
}

ok!(Test {
    p1, k0,
}
);
    "#]],
    );
}

#[test]
fn test_repeat_bad_var() {
    // FIXME: the second rule of the macro should be removed and an error about
    // `$( $c )+` raised
    check(
        r#"
macro_rules! foo {
    ($( $b:ident )+) => { ok!($( $c )+); };
    ($( $b:ident )+) => { ok!($( $b )+); }
}

foo!(b0 b1);
"#,
        expect![[r#"
macro_rules! foo {
    ($( $b:ident )+) => { ok!($( $c )+); };
    ($( $b:ident )+) => { ok!($( $b )+); }
}

ok!(b0 b1);
"#]],
    );
}

#[test]
fn test_issue_3861() {
    // This is should (and does) produce a parse error. It used to infinite loop
    // instead.
    check(
        r#"
macro_rules! rgb_color {
    ($p:expr, $t:ty) => {
        pub fn new() {
            let _ = 0 as $t << $p;
        }
    };
}
// +tree +errors
rgb_color!(8 + 8, u32);
"#,
        expect![[r#"
macro_rules! rgb_color {
    ($p:expr, $t:ty) => {
        pub fn new() {
            let _ = 0 as $t << $p;
        }
    };
}
/* parse error: expected type */
/* parse error: expected R_PAREN */
/* parse error: expected R_ANGLE */
/* parse error: expected COMMA */
/* parse error: expected R_ANGLE */
/* parse error: expected SEMICOLON */
/* parse error: expected expression, item or let statement */
pub fn new() {
    let _ = 0as u32<<(8+8);
}
// MACRO_ITEMS@0..31
//   FN@0..31
//     VISIBILITY@0..3
//       PUB_KW@0..3 "pub"
//     FN_KW@3..5 "fn"
//     NAME@5..8
//       IDENT@5..8 "new"
//     PARAM_LIST@8..10
//       L_PAREN@8..9 "("
//       R_PAREN@9..10 ")"
//     BLOCK_EXPR@10..31
//       STMT_LIST@10..31
//         L_CURLY@10..11 "{"
//         LET_STMT@11..28
//           LET_KW@11..14 "let"
//           WILDCARD_PAT@14..15
//             UNDERSCORE@14..15 "_"
//           EQ@15..16 "="
//           CAST_EXPR@16..28
//             LITERAL@16..17
//               INT_NUMBER@16..17 "0"
//             AS_KW@17..19 "as"
//             PATH_TYPE@19..28
//               PATH@19..28
//                 PATH_SEGMENT@19..28
//                   NAME_REF@19..22
//                     IDENT@19..22 "u32"
//                   GENERIC_ARG_LIST@22..28
//                     L_ANGLE@22..23 "<"
//                     TYPE_ARG@23..27
//                       DYN_TRAIT_TYPE@23..27
//                         TYPE_BOUND_LIST@23..27
//                           TYPE_BOUND@23..26
//                             PATH_TYPE@23..26
//                               PATH@23..26
//                                 PATH_SEGMENT@23..26
//                                   L_ANGLE@23..24 "<"
//                                   PAREN_TYPE@24..26
//                                     L_PAREN@24..25 "("
//                                     ERROR@25..26
//                                       INT_NUMBER@25..26 "8"
//                           PLUS@26..27 "+"
//                     CONST_ARG@27..28
//                       LITERAL@27..28
//                         INT_NUMBER@27..28 "8"
//         ERROR@28..29
//           R_PAREN@28..29 ")"
//         SEMICOLON@29..30 ";"
//         R_CURLY@30..31 "}"

"#]],
    );
}

#[test]
fn test_no_space_after_semi_colon() {
    check(
        r#"
macro_rules! with_std {
    ($($i:item)*) => ($(#[cfg(feature = "std")]$i)*)
}

with_std! {mod m;mod f;}
"#,
        expect![[r##"
macro_rules! with_std {
    ($($i:item)*) => ($(#[cfg(feature = "std")]$i)*)
}

#[cfg(feature = "std")] mod m;
#[cfg(feature = "std")] mod f;
"##]],
    )
}
