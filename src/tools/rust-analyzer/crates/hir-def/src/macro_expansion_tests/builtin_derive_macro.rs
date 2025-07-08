//! Tests for `builtin_derive_macro.rs` from `hir_expand`.

use expect_test::expect;

use crate::macro_expansion_tests::{check, check_errors};

#[test]
fn test_copy_expand_simple() {
    check(
        r#"
//- minicore: derive, copy
#[derive(Copy)]
struct Foo;
"#,
        expect![[r#"
#[derive(Copy)]
struct Foo;

impl <> $crate::marker::Copy for Foo< > where {}"#]],
    );
}

#[test]
fn test_copy_expand_in_core() {
    check(
        r#"
//- /lib.rs crate:core
#[rustc_builtin_macro]
macro derive {}
#[rustc_builtin_macro]
macro Copy {}
#[derive(Copy)]
struct Foo;
"#,
        expect![[r#"
#[rustc_builtin_macro]
macro derive {}
#[rustc_builtin_macro]
macro Copy {}
#[derive(Copy)]
struct Foo;

impl <> $crate::marker::Copy for Foo< > where {}"#]],
    );
}

#[test]
fn test_copy_expand_with_type_params() {
    check(
        r#"
//- minicore: derive, copy
#[derive(Copy)]
struct Foo<A, B>;
"#,
        expect![[r#"
#[derive(Copy)]
struct Foo<A, B>;

impl <A: $crate::marker::Copy, B: $crate::marker::Copy, > $crate::marker::Copy for Foo<A, B, > where {}"#]],
    );
}

#[test]
fn test_copy_expand_with_lifetimes() {
    // We currently just ignore lifetimes
    check(
        r#"
//- minicore: derive, copy
#[derive(Copy)]
struct Foo<A, B, 'a, 'b>;
"#,
        expect![[r#"
#[derive(Copy)]
struct Foo<A, B, 'a, 'b>;

impl <A: $crate::marker::Copy, B: $crate::marker::Copy, > $crate::marker::Copy for Foo<A, B, > where {}"#]],
    );
}

#[test]
fn test_clone_expand() {
    check(
        r#"
//- minicore: derive, clone
#[derive(Clone)]
enum Command<A, B> {
    Move { x: A, y: B },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
#[derive(Clone)]
enum Command<A, B> {
    Move { x: A, y: B },
    Do(&'static str),
    Jump,
}

impl <A: $crate::clone::Clone, B: $crate::clone::Clone, > $crate::clone::Clone for Command<A, B, > where {
    fn clone(&self ) -> Self {
        match self {
            Command::Move {
                x: x, y: y,
            }
            =>Command::Move {
                x: x.clone(), y: y.clone(),
            }
            , Command::Do(f0, )=>Command::Do(f0.clone(), ), Command::Jump=>Command::Jump,
        }
    }
}"#]],
    );
}

#[test]
fn test_clone_expand_with_associated_types() {
    check(
        r#"
//- minicore: derive, clone
trait Trait {
    type InWc;
    type InFieldQualified;
    type InFieldShorthand;
    type InGenericArg;
}
trait Marker {}
struct Vec<T>(T);

#[derive(Clone)]
struct Foo<T: Trait>
where
    <T as Trait>::InWc: Marker,
{
    qualified: <T as Trait>::InFieldQualified,
    shorthand: T::InFieldShorthand,
    generic: Vec<T::InGenericArg>,
}
"#,
        expect![[r#"
trait Trait {
    type InWc;
    type InFieldQualified;
    type InFieldShorthand;
    type InGenericArg;
}
trait Marker {}
struct Vec<T>(T);

#[derive(Clone)]
struct Foo<T: Trait>
where
    <T as Trait>::InWc: Marker,
{
    qualified: <T as Trait>::InFieldQualified,
    shorthand: T::InFieldShorthand,
    generic: Vec<T::InGenericArg>,
}

impl <T: $crate::clone::Clone, > $crate::clone::Clone for Foo<T, > where <T as Trait>::InWc: Marker, T: Trait, T::InFieldShorthand: $crate::clone::Clone, T::InGenericArg: $crate::clone::Clone, {
    fn clone(&self ) -> Self {
        match self {
            Foo {
                qualified: qualified, shorthand: shorthand, generic: generic,
            }
            =>Foo {
                qualified: qualified.clone(), shorthand: shorthand.clone(), generic: generic.clone(),
            }
            ,
        }
    }
}"#]],
    );
}

#[test]
fn test_clone_expand_with_const_generics() {
    check(
        r#"
//- minicore: derive, clone
#[derive(Clone)]
struct Foo<const X: usize, T>(u32);
"#,
        expect![[r#"
#[derive(Clone)]
struct Foo<const X: usize, T>(u32);

impl <const X: usize, T: $crate::clone::Clone, > $crate::clone::Clone for Foo<X, T, > where {
    fn clone(&self ) -> Self {
        match self {
            Foo(f0, )=>Foo(f0.clone(), ),
        }
    }
}"#]],
    );
}

#[test]
fn test_default_expand() {
    check(
        r#"
//- minicore: derive, default
#[derive(Default)]
struct Foo {
    field1: i32,
    field2: (),
}
#[derive(Default)]
enum Bar {
    Foo(u8),
    #[default]
    Bar,
}
"#,
        expect![[r#"
#[derive(Default)]
struct Foo {
    field1: i32,
    field2: (),
}
#[derive(Default)]
enum Bar {
    Foo(u8),
    #[default]
    Bar,
}

impl <> $crate::default::Default for Foo< > where {
    fn default() -> Self {
        Foo {
            field1: $crate::default::Default::default(), field2: $crate::default::Default::default(),
        }
    }
}
impl <> $crate::default::Default for Bar< > where {
    fn default() -> Self {
        Bar::Bar
    }
}"#]],
    );
}

#[test]
fn test_partial_eq_expand() {
    check(
        r#"
//- minicore: derive, eq
#[derive(PartialEq, Eq)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
#[derive(PartialEq, Eq)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}

impl <> $crate::cmp::PartialEq for Command< > where {
    fn eq(&self , other: &Self ) -> bool {
        match (self , other) {
            (Command::Move {
                x: x_self, y: y_self,
            }
            , Command::Move {
                x: x_other, y: y_other,
            }
            )=>x_self.eq(x_other) && y_self.eq(y_other), (Command::Do(f0_self, ), Command::Do(f0_other, ))=>f0_self.eq(f0_other), (Command::Jump, Command::Jump)=>true , _unused=>false
        }
    }
}
impl <> $crate::cmp::Eq for Command< > where {}"#]],
    );
}

#[test]
fn test_partial_eq_expand_with_derive_const() {
    // FIXME: actually expand with const
    check(
        r#"
//- minicore: derive, eq
#[derive_const(PartialEq, Eq)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
#[derive_const(PartialEq, Eq)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}

impl <> $crate::cmp::PartialEq for Command< > where {
    fn eq(&self , other: &Self ) -> bool {
        match (self , other) {
            (Command::Move {
                x: x_self, y: y_self,
            }
            , Command::Move {
                x: x_other, y: y_other,
            }
            )=>x_self.eq(x_other) && y_self.eq(y_other), (Command::Do(f0_self, ), Command::Do(f0_other, ))=>f0_self.eq(f0_other), (Command::Jump, Command::Jump)=>true , _unused=>false
        }
    }
}
impl <> $crate::cmp::Eq for Command< > where {}"#]],
    );
}

#[test]
fn test_partial_ord_expand() {
    check(
        r#"
//- minicore: derive, ord
#[derive(PartialOrd, Ord)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
#[derive(PartialOrd, Ord)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}

impl <> $crate::cmp::PartialOrd for Command< > where {
    fn partial_cmp(&self , other: &Self ) -> $crate::option::Option<$crate::cmp::Ordering> {
        match $crate::intrinsics::discriminant_value(self ).partial_cmp(&$crate::intrinsics::discriminant_value(other)) {
            $crate::option::Option::Some($crate::cmp::Ordering::Equal)=> {
                match (self , other) {
                    (Command::Move {
                        x: x_self, y: y_self,
                    }
                    , Command::Move {
                        x: x_other, y: y_other,
                    }
                    )=>match x_self.partial_cmp(&x_other) {
                        $crate::option::Option::Some($crate::cmp::Ordering::Equal)=> {
                            match y_self.partial_cmp(&y_other) {
                                $crate::option::Option::Some($crate::cmp::Ordering::Equal)=> {
                                    $crate::option::Option::Some($crate::cmp::Ordering::Equal)
                                }
                                c=>return c,
                            }
                        }
                        c=>return c,
                    }
                    , (Command::Do(f0_self, ), Command::Do(f0_other, ))=>match f0_self.partial_cmp(&f0_other) {
                        $crate::option::Option::Some($crate::cmp::Ordering::Equal)=> {
                            $crate::option::Option::Some($crate::cmp::Ordering::Equal)
                        }
                        c=>return c,
                    }
                    , (Command::Jump, Command::Jump)=>$crate::option::Option::Some($crate::cmp::Ordering::Equal), _unused=>$crate::option::Option::Some($crate::cmp::Ordering::Equal)
                }
            }
            c=>return c,
        }
    }
}
impl <> $crate::cmp::Ord for Command< > where {
    fn cmp(&self , other: &Self ) -> $crate::cmp::Ordering {
        match $crate::intrinsics::discriminant_value(self ).cmp(&$crate::intrinsics::discriminant_value(other)) {
            $crate::cmp::Ordering::Equal=> {
                match (self , other) {
                    (Command::Move {
                        x: x_self, y: y_self,
                    }
                    , Command::Move {
                        x: x_other, y: y_other,
                    }
                    )=>match x_self.cmp(&x_other) {
                        $crate::cmp::Ordering::Equal=> {
                            match y_self.cmp(&y_other) {
                                $crate::cmp::Ordering::Equal=> {
                                    $crate::cmp::Ordering::Equal
                                }
                                c=>return c,
                            }
                        }
                        c=>return c,
                    }
                    , (Command::Do(f0_self, ), Command::Do(f0_other, ))=>match f0_self.cmp(&f0_other) {
                        $crate::cmp::Ordering::Equal=> {
                            $crate::cmp::Ordering::Equal
                        }
                        c=>return c,
                    }
                    , (Command::Jump, Command::Jump)=>$crate::cmp::Ordering::Equal, _unused=>$crate::cmp::Ordering::Equal
                }
            }
            c=>return c,
        }
    }
}"#]],
    );
}

#[test]
fn test_hash_expand() {
    check(
        r#"
//- minicore: derive, hash
use core::hash::Hash;

#[derive(Hash)]
struct Foo {
    x: i32,
    y: u64,
    z: (i32, u64),
}
"#,
        expect![[r#"
use core::hash::Hash;

#[derive(Hash)]
struct Foo {
    x: i32,
    y: u64,
    z: (i32, u64),
}

impl <> $crate::hash::Hash for Foo< > where {
    fn hash<H: $crate::hash::Hasher>(&self , ra_expand_state: &mut H) {
        match self {
            Foo {
                x: x, y: y, z: z,
            }
            => {
                x.hash(ra_expand_state);
                y.hash(ra_expand_state);
                z.hash(ra_expand_state);
            }
            ,
        }
    }
}"#]],
    );
    check(
        r#"
//- minicore: derive, hash
use core::hash::Hash;

#[derive(Hash)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
use core::hash::Hash;

#[derive(Hash)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}

impl <> $crate::hash::Hash for Command< > where {
    fn hash<H: $crate::hash::Hasher>(&self , ra_expand_state: &mut H) {
        $crate::mem::discriminant(self ).hash(ra_expand_state);
        match self {
            Command::Move {
                x: x, y: y,
            }
            => {
                x.hash(ra_expand_state);
                y.hash(ra_expand_state);
            }
            , Command::Do(f0, )=> {
                f0.hash(ra_expand_state);
            }
            , Command::Jump=> {}
            ,
        }
    }
}"#]],
    );
}

#[test]
fn test_debug_expand() {
    check(
        r#"
//- minicore: derive, fmt
use core::fmt::Debug;

#[derive(Debug)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
use core::fmt::Debug;

#[derive(Debug)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}

impl <> $crate::fmt::Debug for Command< > where {
    fn fmt(&self , f: &mut $crate::fmt::Formatter) -> $crate::fmt::Result {
        match self {
            Command::Move {
                x: x, y: y,
            }
            =>f.debug_struct("Move").field("x", &x).field("y", &y).finish(), Command::Do(f0, )=>f.debug_tuple("Do").field(&f0).finish(), Command::Jump=>f.write_str("Jump"),
        }
    }
}"#]],
    );
}
#[test]
fn test_debug_expand_with_cfg() {
    check(
        r#"
            //- minicore: derive, fmt
            use core::fmt::Debug;

            #[derive(Debug)]
            struct HideAndShow {
                #[cfg(never)]
                always_hide: u32,
                #[cfg(not(never))]
                always_show: u32,
            }
            #[derive(Debug)]
            enum HideAndShowEnum {
                #[cfg(never)]
                AlwaysHide,
                #[cfg(not(never))]
                AlwaysShow{
                    #[cfg(never)]
                    always_hide: u32,
                    #[cfg(not(never))]
                    always_show: u32,
                }
            }
        "#,
        expect![[r#"
use core::fmt::Debug;

#[derive(Debug)]
struct HideAndShow {
    #[cfg(never)]
    always_hide: u32,
    #[cfg(not(never))]
    always_show: u32,
}
#[derive(Debug)]
enum HideAndShowEnum {
    #[cfg(never)]
    AlwaysHide,
    #[cfg(not(never))]
    AlwaysShow{
        #[cfg(never)]
        always_hide: u32,
        #[cfg(not(never))]
        always_show: u32,
    }
}

impl <> $crate::fmt::Debug for HideAndShow< > where {
    fn fmt(&self , f: &mut $crate::fmt::Formatter) -> $crate::fmt::Result {
        match self {
            HideAndShow {
                always_show: always_show,
            }
            =>f.debug_struct("HideAndShow").field("always_show", &always_show).finish()
        }
    }
}
impl <> $crate::fmt::Debug for HideAndShowEnum< > where {
    fn fmt(&self , f: &mut $crate::fmt::Formatter) -> $crate::fmt::Result {
        match self {
            HideAndShowEnum::AlwaysShow {
                always_show: always_show,
            }
            =>f.debug_struct("AlwaysShow").field("always_show", &always_show).finish(),
        }
    }
}"#]],
    );
}
#[test]
fn test_default_expand_with_cfg() {
    check(
        r#"
//- minicore: derive, default
#[derive(Default)]
struct Foo {
    field1: i32,
    #[cfg(never)]
    field2: (),
    #[cfg(feature = "never")]
    field3: (),
    #[cfg(not(feature = "never"))]
    field4: (),
}
#[derive(Default)]
enum Bar {
    Foo,
    #[cfg_attr(not(never), default)]
    Bar,
}
"#,
        expect![[r##"
#[derive(Default)]
struct Foo {
    field1: i32,
    #[cfg(never)]
    field2: (),
    #[cfg(feature = "never")]
    field3: (),
    #[cfg(not(feature = "never"))]
    field4: (),
}
#[derive(Default)]
enum Bar {
    Foo,
    #[cfg_attr(not(never), default)]
    Bar,
}

impl <> $crate::default::Default for Foo< > where {
    fn default() -> Self {
        Foo {
            field1: $crate::default::Default::default(), field4: $crate::default::Default::default(),
        }
    }
}
impl <> $crate::default::Default for Bar< > where {
    fn default() -> Self {
        Bar::Bar
    }
}"##]],
    );
}

#[test]
fn coerce_pointee_expansion() {
    check(
        r#"
//- minicore: coerce_pointee

use core::marker::CoercePointee;

pub trait Trait<T: ?Sized> {}

#[derive(CoercePointee)]
#[repr(transparent)]
pub struct Foo<'a, T: ?Sized + Trait<U>, #[pointee] U: ?Sized, const N: u32>(T)
where
    U: Trait<U> + ToString;"#,
        expect![[r#"

use core::marker::CoercePointee;

pub trait Trait<T: ?Sized> {}

#[derive(CoercePointee)]
#[repr(transparent)]
pub struct Foo<'a, T: ?Sized + Trait<U>, #[pointee] U: ?Sized, const N: u32>(T)
where
    U: Trait<U> + ToString;
impl <T, U, const N: u32, __S> $crate::ops::DispatchFromDyn<Foo<'a, T, __S, N>> for Foo<T, U, N, > where U: Trait<U> +ToString, T: Trait<__S>, __S: ?Sized, __S: Trait<__S> +ToString, U: ::core::marker::Unsize<__S>, T:?Sized+Trait<U>, U:?Sized, {}
impl <T, U, const N: u32, __S> $crate::ops::CoerceUnsized<Foo<'a, T, __S, N>> for Foo<T, U, N, > where U: Trait<U> +ToString, T: Trait<__S>, __S: ?Sized, __S: Trait<__S> +ToString, U: ::core::marker::Unsize<__S>, T:?Sized+Trait<U>, U:?Sized, {}"#]],
    );
}

#[test]
fn coerce_pointee_errors() {
    check_errors(
        r#"
//- minicore: coerce_pointee

use core::marker::CoercePointee;

#[derive(CoercePointee)]
enum Enum {}

#[derive(CoercePointee)]
struct Struct1;

#[derive(CoercePointee)]
struct Struct2();

#[derive(CoercePointee)]
struct Struct3 {}

#[derive(CoercePointee)]
struct Struct4<T: ?Sized>(T);

#[derive(CoercePointee)]
#[repr(transparent)]
struct Struct5(i32);

#[derive(CoercePointee)]
#[repr(transparent)]
struct Struct6<#[pointee] T: ?Sized, #[pointee] U: ?Sized>(T, U);

#[derive(CoercePointee)]
#[repr(transparent)]
struct Struct7<T: ?Sized, U: ?Sized>(T, U);

#[derive(CoercePointee)]
#[repr(transparent)]
struct Struct8<#[pointee] T, U: ?Sized>(T);

#[derive(CoercePointee)]
#[repr(transparent)]
struct Struct9<T>(T);

#[derive(CoercePointee)]
#[repr(transparent)]
struct Struct9<#[pointee] T, U>(T) where T: ?Sized;
"#,
        expect![[r#"
            35..72: `CoercePointee` can only be derived on `struct`s
            74..114: `CoercePointee` can only be derived on `struct`s with at least one field
            116..158: `CoercePointee` can only be derived on `struct`s with at least one field
            160..202: `CoercePointee` can only be derived on `struct`s with at least one field
            204..258: `CoercePointee` can only be derived on `struct`s with `#[repr(transparent)]`
            260..326: `CoercePointee` can only be derived on `struct`s that are generic over at least one type
            328..439: only one type parameter can be marked as `#[pointee]` when deriving `CoercePointee` traits
            441..530: exactly one generic type parameter must be marked as `#[pointee]` to derive `CoercePointee` traits
            532..621: `derive(CoercePointee)` requires `T` to be marked `?Sized`
            623..690: `derive(CoercePointee)` requires `T` to be marked `?Sized`"#]],
    );
}

#[test]
fn union_derive() {
    check_errors(
        r#"
//- minicore: clone, copy, default, fmt, hash, ord, eq, derive

#[derive(Copy)]
union Foo1 { _v: () }
#[derive(Clone)]
union Foo2 { _v: () }
#[derive(Default)]
union Foo3 { _v: () }
#[derive(Debug)]
union Foo4 { _v: () }
#[derive(Hash)]
union Foo5 { _v: () }
#[derive(Ord)]
union Foo6 { _v: () }
#[derive(PartialOrd)]
union Foo7 { _v: () }
#[derive(Eq)]
union Foo8 { _v: () }
#[derive(PartialEq)]
union Foo9 { _v: () }
    "#,
        expect![[r#"
            78..118: this trait cannot be derived for unions
            119..157: this trait cannot be derived for unions
            158..195: this trait cannot be derived for unions
            196..232: this trait cannot be derived for unions
            233..276: this trait cannot be derived for unions
            313..355: this trait cannot be derived for unions"#]],
    );
}

#[test]
fn default_enum_without_default_attr() {
    check_errors(
        r#"
//- minicore: default, derive

#[derive(Default)]
enum Foo {
    Bar,
}
    "#,
        expect!["1..41: `#[derive(Default)]` on enum with no `#[default]`"],
    );
}

#[test]
fn generic_enum_default() {
    check(
        r#"
//- minicore: default, derive

#[derive(Default)]
enum Foo<T> {
    Bar(T),
    #[default]
    Baz,
}
"#,
        expect![[r#"

#[derive(Default)]
enum Foo<T> {
    Bar(T),
    #[default]
    Baz,
}

impl <T, > $crate::default::Default for Foo<T, > where {
    fn default() -> Self {
        Foo::Baz
    }
}"#]],
    );
}
