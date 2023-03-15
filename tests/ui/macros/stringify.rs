// run-pass
// edition:2021
// compile-flags: --test

#![feature(async_closure)]
#![feature(box_patterns)]
#![feature(const_trait_impl)]
#![feature(decl_macro)]
#![feature(generators)]
#![feature(more_qualified_paths)]
#![feature(raw_ref_op)]
#![feature(trait_alias)]
#![feature(try_blocks)]
#![feature(type_ascription)]
#![deny(unused_macros)]

macro_rules! stringify_block {
    ($block:block) => {
        stringify!($block)
    };
}

macro_rules! stringify_expr {
    ($expr:expr) => {
        stringify!($expr)
    };
}

macro_rules! stringify_item {
    ($item:item) => {
        stringify!($item)
    };
}

macro_rules! stringify_meta {
    ($meta:meta) => {
        stringify!($meta)
    };
}

macro_rules! stringify_pat {
    ($pat:pat) => {
        stringify!($pat)
    };
}

macro_rules! stringify_path {
    ($path:path) => {
        stringify!($path)
    };
}

macro_rules! stringify_stmt {
    ($stmt:stmt) => {
        stringify!($stmt)
    };
}

macro_rules! stringify_ty {
    ($ty:ty) => {
        stringify!($ty)
    };
}

macro_rules! stringify_vis {
    ($vis:vis) => {
        stringify!($vis)
    };
}

#[test]
fn test_block() {
    assert_eq!(stringify_block!({}), "{}");
    assert_eq!(stringify_block!({ true }), "{ true }");
    assert_eq!(stringify_block!({ return }), "{ return }");
    assert_eq!(
        stringify_block!({
            return;
        }),
        "{ return; }",
    );
    assert_eq!(
        stringify_block!({
            let _;
            true
        }),
        "{ let _; true }",
    );
}

#[test]
fn test_expr() {
    // ExprKind::Array
    assert_eq!(stringify_expr!([]), "[]");
    assert_eq!(stringify_expr!([true]), "[true]");
    assert_eq!(stringify_expr!([true,]), "[true]");
    assert_eq!(stringify_expr!([true, true]), "[true, true]");

    // ExprKind::Call
    assert_eq!(stringify_expr!(f()), "f()");
    assert_eq!(stringify_expr!(f::<u8>()), "f::<u8>()");
    assert_eq!(stringify_expr!(f::<1>()), "f::<1>()");
    assert_eq!(stringify_expr!(f::<'a, u8, 1>()), "f::<'a, u8, 1>()");
    assert_eq!(stringify_expr!(f(true)), "f(true)");
    assert_eq!(stringify_expr!(f(true,)), "f(true)");
    assert_eq!(stringify_expr!(()()), "()()");

    // ExprKind::MethodCall
    assert_eq!(stringify_expr!(x.f()), "x.f()");
    assert_eq!(stringify_expr!(x.f::<u8>()), "x.f::<u8>()");

    // ExprKind::Tup
    assert_eq!(stringify_expr!(()), "()");
    assert_eq!(stringify_expr!((true,)), "(true,)");
    assert_eq!(stringify_expr!((true, false)), "(true, false)");
    assert_eq!(stringify_expr!((true, false,)), "(true, false)");

    // ExprKind::Binary
    assert_eq!(stringify_expr!(true || false), "true || false");
    assert_eq!(stringify_expr!(true || false && false), "true || false && false");

    // ExprKind::Unary
    assert_eq!(stringify_expr!(*expr), "*expr");
    assert_eq!(stringify_expr!(!expr), "!expr");
    assert_eq!(stringify_expr!(-expr), "-expr");

    // ExprKind::Lit
    assert_eq!(stringify_expr!('x'), "'x'");
    assert_eq!(stringify_expr!(1_000_i8), "1_000_i8");
    assert_eq!(stringify_expr!(1.00000000000000001), "1.00000000000000001");

    // ExprKind::Cast
    assert_eq!(stringify_expr!(expr as T), "expr as T");
    assert_eq!(stringify_expr!(expr as T<u8>), "expr as T<u8>");

    // ExprKind::Type
    assert_eq!(stringify_expr!(expr: T), "expr: T");
    assert_eq!(stringify_expr!(expr: T<u8>), "expr: T<u8>");

    // ExprKind::If
    assert_eq!(stringify_expr!(if true {}), "if true {}");
    assert_eq!(
        stringify_expr!(if true {
        } else {
        }),
        "if true {} else {}",
    );
    assert_eq!(
        stringify_expr!(if let true = true {
        } else {
        }),
        "if let true = true {} else {}",
    );
    assert_eq!(
        stringify_expr!(if true {
        } else if false {
        }),
        "if true {} else if false {}",
    );
    assert_eq!(
        stringify_expr!(if true {
        } else if false {
        } else {
        }),
        "if true {} else if false {} else {}",
    );
    assert_eq!(
        stringify_expr!(if true {
            return;
        } else if false {
            0
        } else {
            0
        }),
        "if true { return; } else if false { 0 } else { 0 }",
    );

    // ExprKind::While
    assert_eq!(stringify_expr!(while true {}), "while true {}");
    assert_eq!(stringify_expr!('a: while true {}), "'a: while true {}");
    assert_eq!(stringify_expr!(while let true = true {}), "while let true = true {}");

    // ExprKind::ForLoop
    assert_eq!(stringify_expr!(for _ in x {}), "for _ in x {}");
    assert_eq!(stringify_expr!('a: for _ in x {}), "'a: for _ in x {}");

    // ExprKind::Loop
    assert_eq!(stringify_expr!(loop {}), "loop {}");
    assert_eq!(stringify_expr!('a: loop {}), "'a: loop {}");

    // ExprKind::Match
    assert_eq!(stringify_expr!(match self {}), "match self {}");
    assert_eq!(
        stringify_expr!(match self {
            Ok => 1,
        }),
        "match self { Ok => 1, }",
    );
    assert_eq!(
        stringify_expr!(match self {
            Ok => 1,
            Err => 0,
        }),
        "match self { Ok => 1, Err => 0, }",
    );

    // ExprKind::Closure
    assert_eq!(stringify_expr!(|| {}), "|| {}");
    assert_eq!(stringify_expr!(|x| {}), "|x| {}");
    assert_eq!(stringify_expr!(|x: u8| {}), "|x: u8| {}");
    assert_eq!(stringify_expr!(|| ()), "|| ()");
    assert_eq!(stringify_expr!(move || self), "move || self");
    assert_eq!(stringify_expr!(async || self), "async || self");
    assert_eq!(stringify_expr!(async move || self), "async move || self");
    assert_eq!(stringify_expr!(static || self), "static || self");
    assert_eq!(stringify_expr!(static move || self), "static move || self");
    #[rustfmt::skip] // https://github.com/rust-lang/rustfmt/issues/5149
    assert_eq!(
        stringify_expr!(static async || self),
        "static async || self",
    );
    #[rustfmt::skip] // https://github.com/rust-lang/rustfmt/issues/5149
    assert_eq!(
        stringify_expr!(static async move || self),
        "static async move || self",
    );
    assert_eq!(stringify_expr!(|| -> u8 { self }), "|| -> u8 { self }");
    assert_eq!(stringify_expr!(1 + || {}), "1 + (|| {})"); // ??

    // ExprKind::Block
    assert_eq!(stringify_expr!({}), "{}");
    assert_eq!(stringify_expr!(unsafe {}), "unsafe {}");
    assert_eq!(stringify_expr!('a: {}), "'a: {}");
    assert_eq!(
        stringify_expr!(
            #[attr]
            {}
        ),
        "#[attr] {}",
    );
    assert_eq!(
        stringify_expr!(
            {
                #![attr]
            }
        ),
        "{\n\
        \x20   #![attr]\n\
        }",
    );

    // ExprKind::Async
    assert_eq!(stringify_expr!(async {}), "async {}");
    assert_eq!(stringify_expr!(async move {}), "async move {}");

    // ExprKind::Await
    assert_eq!(stringify_expr!(expr.await), "expr.await");

    // ExprKind::TryBlock
    assert_eq!(stringify_expr!(try {}), "try {}");

    // ExprKind::Assign
    assert_eq!(stringify_expr!(expr = true), "expr = true");

    // ExprKind::AssignOp
    assert_eq!(stringify_expr!(expr += true), "expr += true");

    // ExprKind::Field
    assert_eq!(stringify_expr!(expr.field), "expr.field");
    assert_eq!(stringify_expr!(expr.0), "expr.0");

    // ExprKind::Index
    assert_eq!(stringify_expr!(expr[true]), "expr[true]");

    // ExprKind::Range
    assert_eq!(stringify_expr!(..), "..");
    assert_eq!(stringify_expr!(..hi), "..hi");
    assert_eq!(stringify_expr!(lo..), "lo..");
    assert_eq!(stringify_expr!(lo..hi), "lo..hi");
    assert_eq!(stringify_expr!(..=hi), "..=hi");
    assert_eq!(stringify_expr!(lo..=hi), "lo..=hi");
    assert_eq!(stringify_expr!(-2..=-1), "-2..=-1");

    // ExprKind::Path
    assert_eq!(stringify_expr!(thing), "thing");
    assert_eq!(stringify_expr!(m::thing), "m::thing");
    assert_eq!(stringify_expr!(self::thing), "self::thing");
    assert_eq!(stringify_expr!(crate::thing), "crate::thing");
    assert_eq!(stringify_expr!(Self::thing), "Self::thing");
    assert_eq!(stringify_expr!(<Self as T>::thing), "<Self as T>::thing");
    assert_eq!(stringify_expr!(Self::<'static>), "Self::<'static>");

    // ExprKind::AddrOf
    assert_eq!(stringify_expr!(&expr), "&expr");
    assert_eq!(stringify_expr!(&mut expr), "&mut expr");
    assert_eq!(stringify_expr!(&raw const expr), "&raw const expr");
    assert_eq!(stringify_expr!(&raw mut expr), "&raw mut expr");

    // ExprKind::Break
    assert_eq!(stringify_expr!(break), "break");
    assert_eq!(stringify_expr!(break 'a), "break 'a");
    assert_eq!(stringify_expr!(break true), "break true");
    assert_eq!(stringify_expr!(break 'a true), "break 'a true");

    // ExprKind::Continue
    assert_eq!(stringify_expr!(continue), "continue");
    assert_eq!(stringify_expr!(continue 'a), "continue 'a");

    // ExprKind::Ret
    assert_eq!(stringify_expr!(return), "return");
    assert_eq!(stringify_expr!(return true), "return true");

    // ExprKind::MacCall
    assert_eq!(stringify_expr!(mac!(...)), "mac!(...)");
    assert_eq!(stringify_expr!(mac![...]), "mac![...]");
    assert_eq!(stringify_expr!(mac! { ... }), "mac! { ... }");

    // ExprKind::Struct
    assert_eq!(stringify_expr!(Struct {}), "Struct {}");
    #[rustfmt::skip] // https://github.com/rust-lang/rustfmt/issues/5151
    assert_eq!(stringify_expr!(<Struct as Trait>::Type {}), "<Struct as Trait>::Type {}");
    assert_eq!(stringify_expr!(Struct { .. }), "Struct { .. }");
    assert_eq!(stringify_expr!(Struct { ..base }), "Struct { ..base }");
    assert_eq!(stringify_expr!(Struct { x }), "Struct { x }");
    assert_eq!(stringify_expr!(Struct { x, .. }), "Struct { x, .. }");
    assert_eq!(stringify_expr!(Struct { x, ..base }), "Struct { x, ..base }");
    assert_eq!(stringify_expr!(Struct { x: true }), "Struct { x: true }");
    assert_eq!(stringify_expr!(Struct { x: true, .. }), "Struct { x: true, .. }");
    assert_eq!(stringify_expr!(Struct { x: true, ..base }), "Struct { x: true, ..base }");

    // ExprKind::Repeat
    assert_eq!(stringify_expr!([(); 0]), "[(); 0]");

    // ExprKind::Paren
    assert_eq!(stringify_expr!((expr)), "(expr)");

    // ExprKind::Try
    assert_eq!(stringify_expr!(expr?), "expr?");

    // ExprKind::Yield
    assert_eq!(stringify_expr!(yield), "yield");
    assert_eq!(stringify_expr!(yield true), "yield true");
}

#[test]
fn test_item() {
    // ItemKind::ExternCrate
    assert_eq!(
        stringify_item!(
            extern crate std;
        ),
        "extern crate std;",
    );
    assert_eq!(
        stringify_item!(
            pub extern crate self as std;
        ),
        "pub extern crate self as std;",
    );

    // ItemKind::Use
    assert_eq!(
        stringify_item!(
            pub use crate::{a, b::c};
        ),
        "pub use crate::{a, b::c};",
    );

    // ItemKind::Static
    assert_eq!(
        stringify_item!(
            pub static S: () = {};
        ),
        "pub static S: () = {};",
    );
    assert_eq!(
        stringify_item!(
            static mut S: () = {};
        ),
        "static mut S: () = {};",
    );
    assert_eq!(
        stringify_item!(
            static S: ();
        ),
        "static S: ();",
    );
    assert_eq!(
        stringify_item!(
            static mut S: ();
        ),
        "static mut S: ();",
    );

    // ItemKind::Const
    assert_eq!(
        stringify_item!(
            pub const S: () = {};
        ),
        "pub const S: () = {};",
    );
    assert_eq!(
        stringify_item!(
            const S: ();
        ),
        "const S: ();",
    );

    // ItemKind::Fn
    assert_eq!(
        stringify_item!(
            pub default const async unsafe extern "C" fn f() {}
        ),
        "pub default const async unsafe extern \"C\" fn f() {}",
    );

    // ItemKind::Mod
    assert_eq!(
        stringify_item!(
            pub mod m;
        ),
        "pub mod m;",
    );
    assert_eq!(
        stringify_item!(
            mod m {}
        ),
        "mod m {}",
    );
    assert_eq!(
        stringify_item!(
            unsafe mod m;
        ),
        "unsafe mod m;",
    );
    assert_eq!(
        stringify_item!(
            unsafe mod m {}
        ),
        "unsafe mod m {}",
    );

    // ItemKind::ForeignMod
    assert_eq!(
        stringify_item!(
            extern "C" {}
        ),
        "extern \"C\" {}",
    );
    #[rustfmt::skip]
    assert_eq!(
        stringify_item!(
            pub extern "C" {}
        ),
        "extern \"C\" {}",
    );
    assert_eq!(
        stringify_item!(
            unsafe extern "C++" {}
        ),
        "unsafe extern \"C++\" {}",
    );

    // ItemKind::TyAlias
    #[rustfmt::skip]
    assert_eq!(
        stringify_item!(
            pub default type Type<'a>: Bound
            where
                Self: 'a,
            = T;
        ),
        "pub default type Type<'a>: Bound where Self: 'a = T;",
    );

    // ItemKind::Enum
    assert_eq!(
        stringify_item!(
            pub enum Void {}
        ),
        "pub enum Void {}",
    );
    assert_eq!(
        stringify_item!(
            enum Empty {
                Unit,
                Tuple(),
                Struct {},
            }
        ),
        "enum Empty { Unit, Tuple(), Struct {}, }",
    );
    assert_eq!(
        stringify_item!(
            enum Enum<T>
            where
                T: 'a,
            {
                Unit,
                Tuple(T),
                Struct { t: T },
            }
        ),
        "enum Enum<T> where T: 'a {\n\
        \x20   Unit,\n\
        \x20   Tuple(T),\n\
        \x20   Struct {\n\
        \x20       t: T,\n\
        \x20   },\n\
        }",
    );

    // ItemKind::Struct
    assert_eq!(
        stringify_item!(
            pub struct Unit;
        ),
        "pub struct Unit;",
    );
    assert_eq!(
        stringify_item!(
            struct Tuple();
        ),
        "struct Tuple();",
    );
    assert_eq!(
        stringify_item!(
            struct Tuple(T);
        ),
        "struct Tuple(T);",
    );
    assert_eq!(
        stringify_item!(
            struct Struct {}
        ),
        "struct Struct {}",
    );
    assert_eq!(
        stringify_item!(
            struct Struct<T>
            where
                T: 'a,
            {
                t: T,
            }
        ),
        "struct Struct<T> where T: 'a {\n\
        \x20   t: T,\n\
        }",
    );

    // ItemKind::Union
    assert_eq!(
        stringify_item!(
            pub union Union {}
        ),
        "pub union Union {}",
    );
    assert_eq!(
        stringify_item!(
            union Union<T> where T: 'a {
                t: T,
            }
        ),
        "union Union<T> where T: 'a {\n\
        \x20   t: T,\n\
        }",
    );

    // ItemKind::Trait
    assert_eq!(
        stringify_item!(
            pub unsafe auto trait Send {}
        ),
        "pub unsafe auto trait Send {}",
    );
    assert_eq!(
        stringify_item!(
            trait Trait<'a>: Sized
            where
                Self: 'a,
            {
            }
        ),
        "trait Trait<'a>: Sized where Self: 'a {}",
    );

    // ItemKind::TraitAlias
    assert_eq!(
        stringify_item!(
            pub trait Trait<T> = Sized where T: 'a;
        ),
        "pub trait Trait<T> = Sized where T: 'a;",
    );

    // ItemKind::Impl
    assert_eq!(
        stringify_item!(
            pub impl Struct {}
        ),
        "pub impl Struct {}",
    );
    assert_eq!(
        stringify_item!(
            impl<T> Struct<T> {}
        ),
        "impl<T> Struct<T> {}",
    );
    assert_eq!(
        stringify_item!(
            pub impl Trait for Struct {}
        ),
        "pub impl Trait for Struct {}",
    );
    assert_eq!(
        stringify_item!(
            impl<T> const Trait for T {}
        ),
        "impl<T> const Trait for T {}",
    );
    assert_eq!(
        stringify_item!(
            impl ~const Struct {}
        ),
        "impl ~const Struct {}",
    );

    // ItemKind::MacCall
    assert_eq!(stringify_item!(mac!(...);), "mac!(...);");
    assert_eq!(stringify_item!(mac![...];), "mac![...];");
    assert_eq!(stringify_item!(mac! { ... }), "mac! { ... }");

    // ItemKind::MacroDef
    assert_eq!(
        stringify_item!(
            macro_rules! stringify {
                () => {};
            }
        ),
        "macro_rules! stringify { () => {} ; }", // FIXME
    );
    assert_eq!(
        stringify_item!(
            pub macro stringify() {}
        ),
        "pub macro stringify { () => {} }",
    );
}

#[test]
fn test_meta() {
    assert_eq!(stringify_meta!(k), "k");
    assert_eq!(stringify_meta!(k = "v"), "k = \"v\"");
    assert_eq!(stringify_meta!(list(k1, k2 = "v")), "list(k1, k2 = \"v\")");
    assert_eq!(stringify_meta!(serde::k), "serde::k");
}

#[test]
fn test_pat() {
    // PatKind::Wild
    assert_eq!(stringify_pat!(_), "_");

    // PatKind::Ident
    assert_eq!(stringify_pat!(_x), "_x");
    assert_eq!(stringify_pat!(ref _x), "ref _x");
    assert_eq!(stringify_pat!(mut _x), "mut _x");
    assert_eq!(stringify_pat!(ref mut _x), "ref mut _x");
    assert_eq!(stringify_pat!(ref mut _x @ _), "ref mut _x @ _");

    // PatKind::Struct
    assert_eq!(stringify_pat!(Struct {}), "Struct {}");
    assert_eq!(stringify_pat!(Struct::<u8> {}), "Struct::<u8> {}");
    assert_eq!(stringify_pat!(Struct::<'static> {}), "Struct::<'static> {}");
    assert_eq!(stringify_pat!(Struct { x }), "Struct { x }");
    assert_eq!(stringify_pat!(Struct { x: _x }), "Struct { x: _x }");
    assert_eq!(stringify_pat!(Struct { .. }), "Struct { .. }");
    assert_eq!(stringify_pat!(Struct { x, .. }), "Struct { x, .. }");
    assert_eq!(stringify_pat!(Struct { x: _x, .. }), "Struct { x: _x, .. }");
    #[rustfmt::skip] // https://github.com/rust-lang/rustfmt/issues/5151
    assert_eq!(
        stringify_pat!(<Struct as Trait>::Type {}),
        "<Struct as Trait>::Type {}",
    );

    // PatKind::TupleStruct
    assert_eq!(stringify_pat!(Tuple()), "Tuple()");
    assert_eq!(stringify_pat!(Tuple::<u8>()), "Tuple::<u8>()");
    assert_eq!(stringify_pat!(Tuple::<'static>()), "Tuple::<'static>()");
    assert_eq!(stringify_pat!(Tuple(x)), "Tuple(x)");
    assert_eq!(stringify_pat!(Tuple(..)), "Tuple(..)");
    assert_eq!(stringify_pat!(Tuple(x, ..)), "Tuple(x, ..)");
    assert_eq!(stringify_pat!(<Struct as Trait>::Type()), "<Struct as Trait>::Type()");

    // PatKind::Or
    assert_eq!(stringify_pat!(true | false), "true | false");
    assert_eq!(stringify_pat!(| true), "true");
    assert_eq!(stringify_pat!(|true| false), "true | false");

    // PatKind::Path
    assert_eq!(stringify_pat!(crate::Path), "crate::Path");
    assert_eq!(stringify_pat!(Path::<u8>), "Path::<u8>");
    assert_eq!(stringify_pat!(Path::<'static>), "Path::<'static>");
    assert_eq!(stringify_pat!(<Struct as Trait>::Type), "<Struct as Trait>::Type");

    // PatKind::Tuple
    assert_eq!(stringify_pat!(()), "()");
    assert_eq!(stringify_pat!((true,)), "(true,)");
    assert_eq!(stringify_pat!((true, false)), "(true, false)");

    // PatKind::Box
    assert_eq!(stringify_pat!(box pat), "box pat");

    // PatKind::Ref
    assert_eq!(stringify_pat!(&pat), "&pat");
    assert_eq!(stringify_pat!(&mut pat), "&mut pat");

    // PatKind::Lit
    assert_eq!(stringify_pat!(1_000_i8), "1_000_i8");

    // PatKind::Range
    assert_eq!(stringify_pat!(..1), "..1");
    assert_eq!(stringify_pat!(0..), "0..");
    assert_eq!(stringify_pat!(0..1), "0..1");
    assert_eq!(stringify_pat!(0..=1), "0..=1");
    assert_eq!(stringify_pat!(-2..=-1), "-2..=-1");

    // PatKind::Slice
    assert_eq!(stringify_pat!([]), "[]");
    assert_eq!(stringify_pat!([true]), "[true]");
    assert_eq!(stringify_pat!([true,]), "[true]");
    assert_eq!(stringify_pat!([true, false]), "[true, false]");

    // PatKind::Rest
    assert_eq!(stringify_pat!(..), "..");

    // PatKind::Paren
    assert_eq!(stringify_pat!((pat)), "(pat)");

    // PatKind::MacCall
    assert_eq!(stringify_pat!(mac!(...)), "mac!(...)");
    assert_eq!(stringify_pat!(mac![...]), "mac![...]");
    assert_eq!(stringify_pat!(mac! { ... }), "mac! { ... }");
}

#[test]
fn test_path() {
    assert_eq!(stringify_path!(thing), "thing");
    assert_eq!(stringify_path!(m::thing), "m::thing");
    assert_eq!(stringify_path!(self::thing), "self::thing");
    assert_eq!(stringify_path!(crate::thing), "crate::thing");
    assert_eq!(stringify_path!(Self::thing), "Self::thing");
    assert_eq!(stringify_path!(Self<'static>), "Self<'static>");
    assert_eq!(stringify_path!(Self::<'static>), "Self<'static>");
    assert_eq!(stringify_path!(Self()), "Self()");
    assert_eq!(stringify_path!(Self() -> ()), "Self() -> ()");
}

#[test]
fn test_stmt() {
    // StmtKind::Local
    assert_eq!(stringify_stmt!(let _), "let _;");
    assert_eq!(stringify_stmt!(let x = true), "let x = true;");
    assert_eq!(stringify_stmt!(let x: bool = true), "let x: bool = true;");

    // StmtKind::Item
    assert_eq!(
        stringify_stmt!(
            struct S;
        ),
        "struct S;",
    );

    // StmtKind::Expr
    assert_eq!(stringify_stmt!(loop {}), "loop {}");

    // StmtKind::Semi
    assert_eq!(stringify_stmt!(1 + 1), "1 + 1;");

    // StmtKind::Empty
    assert_eq!(stringify_stmt!(;), ";");

    // StmtKind::MacCall
    assert_eq!(stringify_stmt!(mac!(...)), "mac!(...)");
    assert_eq!(stringify_stmt!(mac![...]), "mac![...]");
    assert_eq!(stringify_stmt!(mac! { ... }), "mac! { ... }");
}

#[test]
fn test_ty() {
    // TyKind::Slice
    assert_eq!(stringify_ty!([T]), "[T]");

    // TyKind::Array
    assert_eq!(stringify_ty!([T; 0]), "[T; 0]");

    // TyKind::Ptr
    assert_eq!(stringify_ty!(*const T), "*const T");
    assert_eq!(stringify_ty!(*mut T), "*mut T");

    // TyKind::Ref
    assert_eq!(stringify_ty!(&T), "&T");
    assert_eq!(stringify_ty!(&mut T), "&mut T");
    assert_eq!(stringify_ty!(&'a T), "&'a T");
    assert_eq!(stringify_ty!(&'a mut T), "&'a mut T");

    // TyKind::BareFn
    assert_eq!(stringify_ty!(fn()), "fn()");
    assert_eq!(stringify_ty!(fn() -> ()), "fn() -> ()");
    assert_eq!(stringify_ty!(fn(u8)), "fn(u8)");
    assert_eq!(stringify_ty!(fn(x: u8)), "fn(x: u8)");
    #[rustfmt::skip]
    assert_eq!(stringify_ty!(for<> fn()), "fn()");
    assert_eq!(stringify_ty!(for<'a> fn()), "for<'a> fn()");

    // TyKind::Never
    assert_eq!(stringify_ty!(!), "!");

    // TyKind::Tup
    assert_eq!(stringify_ty!(()), "()");
    assert_eq!(stringify_ty!((T,)), "(T,)");
    assert_eq!(stringify_ty!((T, U)), "(T, U)");

    // TyKind::Path
    assert_eq!(stringify_ty!(T), "T");
    assert_eq!(stringify_ty!(Ref<'a>), "Ref<'a>");
    assert_eq!(stringify_ty!(PhantomData<T>), "PhantomData<T>");
    assert_eq!(stringify_ty!(PhantomData::<T>), "PhantomData<T>");
    assert_eq!(stringify_ty!(Fn() -> !), "Fn() -> !");
    assert_eq!(stringify_ty!(Fn(u8) -> !), "Fn(u8) -> !");
    assert_eq!(stringify_ty!(<Struct as Trait>::Type), "<Struct as Trait>::Type");

    // TyKind::TraitObject
    assert_eq!(stringify_ty!(dyn Send), "dyn Send");
    assert_eq!(stringify_ty!(dyn Send + 'a), "dyn Send + 'a");
    assert_eq!(stringify_ty!(dyn 'a + Send), "dyn 'a + Send");
    assert_eq!(stringify_ty!(dyn ?Sized), "dyn ?Sized");
    assert_eq!(stringify_ty!(dyn ~const Clone), "dyn ~const Clone");
    assert_eq!(stringify_ty!(dyn for<'a> Send), "dyn for<'a> Send");

    // TyKind::ImplTrait
    assert_eq!(stringify_ty!(impl Send), "impl Send");
    assert_eq!(stringify_ty!(impl Send + 'a), "impl Send + 'a");
    assert_eq!(stringify_ty!(impl 'a + Send), "impl 'a + Send");
    assert_eq!(stringify_ty!(impl ?Sized), "impl ?Sized");
    assert_eq!(stringify_ty!(impl ~const Clone), "impl ~const Clone");
    assert_eq!(stringify_ty!(impl for<'a> Send), "impl for<'a> Send");

    // TyKind::Paren
    assert_eq!(stringify_ty!((T)), "(T)");

    // TyKind::Infer
    assert_eq!(stringify_ty!(_), "_");

    // TyKind::MacCall
    assert_eq!(stringify_ty!(mac!(...)), "mac!(...)");
    assert_eq!(stringify_ty!(mac![...]), "mac![...]");
    assert_eq!(stringify_ty!(mac! { ... }), "mac! { ... }");
}

#[test]
fn test_vis() {
    // VisibilityKind::Public
    assert_eq!(stringify_vis!(pub), "pub ");

    // VisibilityKind::Restricted
    assert_eq!(stringify_vis!(pub(crate)), "pub(crate) ");
    assert_eq!(stringify_vis!(pub(self)), "pub(self) ");
    assert_eq!(stringify_vis!(pub(super)), "pub(super) ");
    assert_eq!(stringify_vis!(pub(in crate)), "pub(in crate) ");
    assert_eq!(stringify_vis!(pub(in self)), "pub(in self) ");
    assert_eq!(stringify_vis!(pub(in super)), "pub(in super) ");
    assert_eq!(stringify_vis!(pub(in path::to)), "pub(in path::to) ");
    assert_eq!(stringify_vis!(pub(in ::path::to)), "pub(in ::path::to) ");
    assert_eq!(stringify_vis!(pub(in self::path::to)), "pub(in self::path::to) ");
    assert_eq!(stringify_vis!(pub(in super::path::to)), "pub(in super::path::to) ");

    // VisibilityKind::Inherited
    // Directly calling `stringify_vis!()` does not work.
    macro_rules! stringify_inherited_vis {
        ($vis:vis struct) => {
            stringify_vis!($vis)
        };
    }
    assert_eq!(stringify_inherited_vis!(struct), "");
}
