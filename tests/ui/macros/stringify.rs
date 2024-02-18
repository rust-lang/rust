//@ run-pass
//@ edition:2021
//@ compile-flags: --test

#![allow(incomplete_features)]
#![feature(async_closure)]
#![feature(auto_traits)]
#![feature(box_patterns)]
#![feature(const_trait_impl)]
#![feature(coroutines)]
#![feature(decl_macro)]
#![feature(explicit_tail_calls)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(more_qualified_paths)]
#![feature(never_patterns)]
#![feature(raw_ref_op)]
#![feature(trait_alias)]
#![feature(try_blocks)]
#![feature(type_ascription)]
#![feature(yeet_expr)]
#![deny(unused_macros)]

// These macros force the use of AST pretty-printing by converting the input to
// a particular fragment specifier.
macro_rules! block { ($block:block) => { stringify!($block) }; }
macro_rules! expr { ($expr:expr) => { stringify!($expr) }; }
macro_rules! item { ($item:item) => { stringify!($item) }; }
macro_rules! meta { ($meta:meta) => { stringify!($meta) }; }
macro_rules! pat { ($pat:pat) => { stringify!($pat) }; }
macro_rules! path { ($path:path) => { stringify!($path) }; }
macro_rules! stmt { ($stmt:stmt) => { stringify!($stmt) }; }
macro_rules! ty { ($ty:ty) => { stringify!($ty) }; }
macro_rules! vis { ($vis:vis) => { stringify!($vis) }; }

// Use this when AST pretty-printing and TokenStream pretty-printing give
// the same result (which is preferable.)
macro_rules! c1 {
    ($frag:ident, [$($tt:tt)*], $s:literal) => {
        assert_eq!($frag!($($tt)*), $s);
        assert_eq!(stringify!($($tt)*), $s);
    };
}

// Use this when AST pretty-printing and TokenStream pretty-printing give
// different results.
//
// `c1` and `c2` could be in a single macro, but having them separate makes it
// easy to find the cases where the two pretty-printing approaches give
// different results.
macro_rules! c2 {
    ($frag:ident, [$($tt:tt)*], $s1:literal, $s2:literal $(,)?) => {
        assert_ne!($s1, $s2, "should use `c1!` instead");
        assert_eq!($frag!($($tt)*), $s1);
        assert_eq!(stringify!($($tt)*), $s2);
    };
}

#[test]
fn test_block() {
    c1!(block, [ {} ], "{}");
    c1!(block, [ { true } ], "{ true }");
    c1!(block, [ { return } ], "{ return }");
    c1!(block, [ { return; } ], "{ return; }");
    c1!(block,
        [ {
            let _;
            true
        } ],
        "{ let _; true }"
    );
}

#[test]
fn test_expr() {
    // ExprKind::Array
    c1!(expr, [ [] ], "[]");
    c1!(expr, [ [true] ], "[true]");
    c2!(expr, [ [true,] ], "[true]", "[true,]");
    c1!(expr, [ [true, true] ], "[true, true]");

    // ExprKind::ConstBlock
    // FIXME: todo

    // ExprKind::Call
    c1!(expr, [ f() ], "f()");
    c1!(expr, [ f::<u8>() ], "f::<u8>()");
    c2!(expr, [ f ::  < u8>( ) ], "f::<u8>()", "f :: < u8>()");
    c1!(expr, [ f::<1>() ], "f::<1>()");
    c1!(expr, [ f::<'a, u8, 1>() ], "f::<'a, u8, 1>()");
    c1!(expr, [ f(true) ], "f(true)");
    c2!(expr, [ f(true,) ], "f(true)", "f(true,)");
    c1!(expr, [ ()() ], "()()");

    // ExprKind::MethodCall
    c1!(expr, [ x.f() ], "x.f()");
    c1!(expr, [ x.f::<u8>() ], "x.f::<u8>()");
    c1!(expr, [ x.collect::<Vec<_>>() ], "x.collect::<Vec<_>>()");

    // ExprKind::Tup
    c1!(expr, [ () ], "()");
    c1!(expr, [ (true,) ], "(true,)");
    c1!(expr, [ (true, false) ], "(true, false)");
    c2!(expr, [ (true, false,) ], "(true, false)", "(true, false,)");

    // ExprKind::Binary
    c1!(expr, [ true || false ], "true || false");
    c1!(expr, [ true || false && false ], "true || false && false");
    c1!(expr, [ a < 1 && 2 < b && c > 3 && 4 > d ], "a < 1 && 2 < b && c > 3 && 4 > d");
    c1!(expr, [ a & b & !c ], "a & b & !c");
    c1!(expr, [ a + b * c - d + -1 * -2 - -3], "a + b * c - d + -1 * -2 - -3");
    c1!(expr, [ x = !y ], "x = !y");

    // ExprKind::Unary
    c1!(expr, [ *expr ], "*expr");
    c1!(expr, [ !expr ], "!expr");
    c1!(expr, [ -expr ], "-expr");

    // ExprKind::Lit
    c1!(expr, [ 'x' ], "'x'");
    c1!(expr, [ 1_000_i8 ], "1_000_i8");
    c1!(expr, [ 1.00000000000000001 ], "1.00000000000000001");

    // ExprKind::Cast
    c1!(expr, [ expr as T ], "expr as T");
    c1!(expr, [ expr as T<u8> ], "expr as T<u8>");

    // ExprKind::Type: there is no syntax for type ascription.

    // ExprKind::Let
    c1!(expr, [ if let Some(a) = b { c } else { d } ], "if let Some(a) = b { c } else { d }");
    c1!(expr, [ if let _ = true && false {} ], "if let _ = true && false {}");
    c1!(expr, [ if let _ = (true && false) {} ], "if let _ = (true && false) {}");
    macro_rules! c2_if_let {
        ($expr:expr, $expr_expected:expr, $tokens_expected:expr $(,)?) => {
            c2!(expr, [ if let _ = $expr {} ], $expr_expected, $tokens_expected);
        };
    }
    c2_if_let!(
        true && false,
        "if let _ = (true && false) {}",
        "if let _ = true && false {}",
    );
    c1!(expr,
        [ match () { _ if let _ = Struct {} => {} } ],
        "match () { _ if let _ = Struct {} => {} }"
    );

    // ExprKind::If
    c1!(expr, [ if true {} ], "if true {}");
    c1!(expr, [ if !true {} ], "if !true {}");
    c1!(expr, [ if ::std::blah() { } else { } ], "if ::std::blah() {} else {}");
    c1!(expr, [ if let true = true {} else {} ], "if let true = true {} else {}");
    c1!(expr,
        [ if true {
        } else if false {
        } ],
        "if true {} else if false {}"
    );
    c1!(expr,
        [ if true {
        } else if false {
        } else {
        } ],
        "if true {} else if false {} else {}"
    );
    c1!(expr,
        [ if true {
            return;
        } else if false {
            0
        } else {
            0
        } ],
        "if true { return; } else if false { 0 } else { 0 }"
    );

    // ExprKind::While
    c1!(expr, [ while true {} ], "while true {}");
    c1!(expr, [ 'a: while true {} ], "'a: while true {}");
    c1!(expr, [ while let true = true {} ], "while let true = true {}");

    // ExprKind::ForLoop
    c1!(expr, [ for _ in x {} ], "for _ in x {}");
    c1!(expr, [ 'a: for _ in x {} ], "'a: for _ in x {}");

    // ExprKind::Loop
    c1!(expr, [ loop {} ], "loop {}");
    c1!(expr, [ 'a: loop {} ], "'a: loop {}");

    // ExprKind::Match
    c1!(expr, [ match self {} ], "match self {}");
    c1!(expr,
        [ match self {
            Ok => 1,
        } ],
        "match self { Ok => 1, }"
    );
    c1!(expr,
        [ match self {
            Ok => 1,
            Err => 0,
        } ],
        "match self { Ok => 1, Err => 0, }"
    );
    macro_rules! c2_match_arm {
        ([ $expr:expr ], $expr_expected:expr, $tokens_expected:expr $(,)?) => {
            c2!(expr, [ match () { _ => $expr } ], $expr_expected, $tokens_expected);
        };
    }
    c2_match_arm!(
        [ { 1 } - 1 ],
        "match () { _ => ({ 1 }) - 1, }",
        "match () { _ => { 1 } - 1 }",
    );

    // ExprKind::Closure
    c1!(expr, [ || {} ], "|| {}");
    c1!(expr, [ |x| {} ], "|x| {}");
    c1!(expr, [ |x: u8| {} ], "|x: u8| {}");
    c1!(expr, [ || () ], "|| ()");
    c1!(expr, [ move || self ], "move || self");
    c1!(expr, [ async || self ], "async || self");
    c1!(expr, [ async move || self ], "async move || self");
    c1!(expr, [ static || self ], "static || self");
    c1!(expr, [ static move || self ], "static move || self");
    c1!(expr, [ static async || self ], "static async || self");
    c1!(expr, [ static async move || self ], "static async move || self");
    c1!(expr, [ || -> u8 { self } ], "|| -> u8 { self }");
    c2!(expr, [ 1 + || {} ], "1 + (|| {})", "1 + || {}"); // AST??

    // ExprKind::Block
    c1!(expr, [ {} ], "{}");
    c1!(expr, [ unsafe {} ], "unsafe {}");
    c1!(expr, [ 'a: {} ], "'a: {}");
    c1!(expr, [ #[attr] {} ], "#[attr] {}");
    c2!(expr,
        [
            {
                #![attr]
            }
        ],
        "{\n\
        \x20   #![attr]\n\
        }",
        "{ #![attr] }"
    );

    // ExprKind::Async
    c1!(expr, [ async {} ], "async {}");
    c1!(expr, [ async move {} ], "async move {}");

    // ExprKind::Await
    c1!(expr, [ expr.await ], "expr.await");

    // ExprKind::TryBlock
    c1!(expr, [ try {} ], "try {}");

    // ExprKind::Assign
    c1!(expr, [ expr = true ], "expr = true");

    // ExprKind::AssignOp
    c1!(expr, [ expr += true ], "expr += true");

    // ExprKind::Field
    c1!(expr, [ expr.field ], "expr.field");
    c1!(expr, [ expr.0 ], "expr.0");

    // ExprKind::Index
    c1!(expr, [ expr[true] ], "expr[true]");

    // ExprKind::Range
    c1!(expr, [ .. ], "..");
    c1!(expr, [ ..hi ], "..hi");
    c1!(expr, [ lo.. ], "lo..");
    c1!(expr, [ lo..hi ], "lo..hi");
    c2!(expr, [ lo .. hi ], "lo..hi", "lo .. hi");
    c1!(expr, [ ..=hi ], "..=hi");
    c1!(expr, [ lo..=hi ], "lo..=hi");
    c1!(expr, [ -2..=-1 ], "-2..=-1");

    // ExprKind::Underscore
    // FIXME: todo

    // ExprKind::Path
    c1!(expr, [ thing ], "thing");
    c1!(expr, [ m::thing ], "m::thing");
    c1!(expr, [ self::thing ], "self::thing");
    c1!(expr, [ crate::thing ], "crate::thing");
    c1!(expr, [ Self::thing ], "Self::thing");
    c1!(expr, [ <Self as T>::thing ], "<Self as T>::thing");
    c1!(expr, [ Self::<'static> ], "Self::<'static>");

    // ExprKind::AddrOf
    c1!(expr, [ &expr ], "&expr");
    c1!(expr, [ &mut expr ], "&mut expr");
    c1!(expr, [ &raw const expr ], "&raw const expr");
    c1!(expr, [ &raw mut expr ], "&raw mut expr");

    // ExprKind::Break
    c1!(expr, [ break ], "break");
    c1!(expr, [ break 'a ], "break 'a");
    c1!(expr, [ break true ], "break true");
    c1!(expr, [ break 'a true ], "break 'a true");

    // ExprKind::Continue
    c1!(expr, [ continue ], "continue");
    c1!(expr, [ continue 'a ], "continue 'a");

    // ExprKind::Ret
    c1!(expr, [ return ], "return");
    c1!(expr, [ return true ], "return true");

    // ExprKind::InlineAsm: untestable because this test works pre-expansion.

    // ExprKind::OffsetOf: untestable because this test works pre-expansion.

    // ExprKind::MacCall
    c1!(expr, [ mac!(...) ], "mac!(...)");
    c1!(expr, [ mac![...] ], "mac![...]");
    c1!(expr, [ mac! { ... } ], "mac! { ... }");

    // ExprKind::Struct
    c1!(expr, [ Struct {} ], "Struct {}");
    c1!(expr, [ <Struct as Trait>::Type {} ], "<Struct as Trait>::Type {}");
    c1!(expr, [ Struct { .. } ], "Struct { .. }");
    c1!(expr, [ Struct { ..base } ], "Struct { ..base }");
    c1!(expr, [ Struct { x } ], "Struct { x }");
    c1!(expr, [ Struct { x, .. } ], "Struct { x, .. }");
    c1!(expr, [ Struct { x, ..base } ], "Struct { x, ..base }");
    c1!(expr, [ Struct { x: true } ], "Struct { x: true }");
    c1!(expr, [ Struct { x: true, .. } ], "Struct { x: true, .. }");
    c1!(expr, [ Struct { x: true, ..base } ], "Struct { x: true, ..base }");

    // ExprKind::Repeat
    c1!(expr, [ [(); 0] ], "[(); 0]");

    // ExprKind::Paren
    c1!(expr, [ (expr) ], "(expr)");

    // ExprKind::Try
    c1!(expr, [ expr? ], "expr?");

    // ExprKind::Yield
    c1!(expr, [ yield ], "yield");
    c1!(expr, [ yield true ], "yield true");

    // ExprKind::Yeet
    c1!(expr, [ do yeet ], "do yeet");
    c1!(expr, [ do yeet 0 ], "do yeet 0");

    // ExprKind::Become
    // FIXME: todo

    // ExprKind::IncludedBytes
    // FIXME: todo

    // ExprKind::FormatArgs: untestable because this test works pre-expansion.

    // ExprKind::Err: untestable.
}

#[test]
fn test_item() {
    // ItemKind::ExternCrate
    c1!(item, [ extern crate std; ], "extern crate std;");
    c1!(item, [ pub extern crate self as std; ], "pub extern crate self as std;");

    // ItemKind::Use
    c2!(item,
        [ pub use crate::{a, b::c}; ],
        "pub use crate::{a, b::c};",
        "pub use crate::{ a, b::c };" // FIXME
    );
    c1!(item, [ pub use A::*; ], "pub use A::*;");

    // ItemKind::Static
    c1!(item, [ pub static S: () = {}; ], "pub static S: () = {};");
    c1!(item, [ static mut S: () = {}; ], "static mut S: () = {};");
    c1!(item, [ static S: (); ], "static S: ();");
    c1!(item, [ static mut S: (); ], "static mut S: ();");

    // ItemKind::Const
    c1!(item, [ pub const S: () = {}; ], "pub const S: () = {};");
    c1!(item, [ const S: (); ], "const S: ();");

    // ItemKind::Fn
    c1!(item,
        [ pub default const async unsafe extern "C" fn f() {} ],
        "pub default const async unsafe extern \"C\" fn f() {}"
    );
    c1!(item, [ fn g<T>(t: Vec<Vec<Vec<T>>>) {} ], "fn g<T>(t: Vec<Vec<Vec<T>>>) {}");
    c1!(item,
        [ fn h<'a>(t: &'a Vec<Cell<dyn D>>) {} ],
        "fn h<'a>(t: &'a Vec<Cell<dyn D>>) {}"
    );

    // ItemKind::Mod
    c1!(item, [ pub mod m; ], "pub mod m;");
    c1!(item, [ mod m {} ], "mod m {}");
    c1!(item, [ unsafe mod m; ], "unsafe mod m;");
    c1!(item, [ unsafe mod m {} ], "unsafe mod m {}");

    // ItemKind::ForeignMod
    c1!(item, [ extern "C" {} ], "extern \"C\" {}");
    c2!(item,
        [ pub extern "C" {} ],
        "extern \"C\" {}", // ??
        "pub extern \"C\" {}"
    );
    c1!(item, [ unsafe extern "C++" {} ], "unsafe extern \"C++\" {}");

    // ItemKind::GlobalAsm: untestable because this test works pre-expansion.

    // ItemKind::TyAlias
    c2!(item,
        [
            pub default type Type<'a>: Bound
            where
                Self: 'a,
            = T;
        ],
        "pub default type Type<'a>: Bound where Self: 'a = T;",
        "pub default type Type<'a>: Bound where Self: 'a, = T;"
    );

    // ItemKind::Enum
    c1!(item, [ pub enum Void {} ], "pub enum Void {}");
    c1!(item,
        [
            enum Empty {
                Unit,
                Tuple(),
                Struct {},
            }
        ],
        "enum Empty { Unit, Tuple(), Struct {}, }"
    );
    c2!(item,
        [
            enum Enum<T>
            where
                T: 'a,
            {
                Unit,
                Tuple(T),
                Struct { t: T },
            }
        ],
        "enum Enum<T> where T: 'a {\n\
        \x20   Unit,\n\
        \x20   Tuple(T),\n\
        \x20   Struct {\n\
        \x20       t: T,\n\
        \x20   },\n\
        }",
        "enum Enum<T> where T: 'a, { Unit, Tuple(T), Struct { t: T }, }"
    );

    // ItemKind::Struct
    c1!(item, [ pub struct Unit; ], "pub struct Unit;");
    c1!(item, [ struct Tuple(); ], "struct Tuple();");
    c1!(item, [ struct Tuple(T); ], "struct Tuple(T);");
    c1!(item, [ struct Struct {} ], "struct Struct {}");
    c2!(item,
        [
            struct Struct<T>
            where
                T: 'a,
            {
                t: T,
            }
        ],
        "struct Struct<T> where T: 'a {\n\
        \x20   t: T,\n\
        }",
        "struct Struct<T> where T: 'a, { t: T, }"
    );

    // ItemKind::Union
    c1!(item, [ pub union Union {} ], "pub union Union {}");
    c2!(item,
        [
            union Union<T> where T: 'a {
                t: T,
            }
        ],
        "union Union<T> where T: 'a {\n\
        \x20   t: T,\n\
        }",
        "union Union<T> where T: 'a { t: T, }"
    );

    // ItemKind::Trait
    c1!(item, [ pub unsafe auto trait Send {} ], "pub unsafe auto trait Send {}");
    c2!(item,
        [
            trait Trait<'a>: Sized
            where
                Self: 'a,
            {
            }
        ],
        "trait Trait<'a>: Sized where Self: 'a {}",
        "trait Trait<'a>: Sized where Self: 'a, {}"
    );

    // ItemKind::TraitAlias
    c1!(item,
        [ pub trait Trait<T> = Sized where T: 'a; ],
        "pub trait Trait<T> = Sized where T: 'a;"
    );

    // ItemKind::Impl
    c1!(item, [ pub impl Struct {} ], "pub impl Struct {}");
    c1!(item, [ impl<T> Struct<T> {} ], "impl<T> Struct<T> {}");
    c1!(item, [ pub impl Trait for Struct {} ], "pub impl Trait for Struct {}");
    c1!(item, [ impl<T> const Trait for T {} ], "impl<T> const Trait for T {}");
    c1!(item, [ impl ~const Struct {} ], "impl ~const Struct {}");

    // ItemKind::MacCall
    c1!(item, [ mac!(...); ], "mac!(...);");
    c1!(item, [ mac![...]; ], "mac![...];");
    c1!(item, [ mac! { ... } ], "mac! { ... }");

    // ItemKind::MacroDef
    c1!(item,
        [
            macro_rules! stringify {
                () => {};
            }
        ],
        "macro_rules! stringify { () => {}; }"
    );
    c2!(item,
        [ pub macro stringify() {} ],
        "pub macro stringify { () => {} }", // ??
        "pub macro stringify() {}"
    );
}

#[test]
fn test_meta() {
    c1!(meta, [ k ], "k");
    c1!(meta, [ k = "v" ], "k = \"v\"");
    c1!(meta, [ list(k1, k2 = "v") ], "list(k1, k2 = \"v\")");
    c1!(meta, [ serde::k ], "serde::k");
}

#[test]
fn test_pat() {
    // PatKind::Wild
    c1!(pat, [ _ ], "_");

    // PatKind::Ident
    c1!(pat, [ _x ], "_x");
    c1!(pat, [ ref _x ], "ref _x");
    c1!(pat, [ mut _x ], "mut _x");
    c1!(pat, [ ref mut _x ], "ref mut _x");
    c1!(pat, [ ref mut _x @ _ ], "ref mut _x @ _");

    // PatKind::Struct
    c1!(pat, [ Struct {} ], "Struct {}");
    c1!(pat, [ Struct::<u8> {} ], "Struct::<u8> {}");
    c2!(pat, [ Struct ::< u8 > {} ], "Struct::<u8> {}", "Struct ::< u8 > {}");
    c1!(pat, [ Struct::<'static> {} ], "Struct::<'static> {}");
    c1!(pat, [ Struct { x } ], "Struct { x }");
    c1!(pat, [ Struct { x: _x } ], "Struct { x: _x }");
    c1!(pat, [ Struct { .. } ], "Struct { .. }");
    c1!(pat, [ Struct { x, .. } ], "Struct { x, .. }");
    c1!(pat, [ Struct { x: _x, .. } ], "Struct { x: _x, .. }");
    c1!(pat, [ <Struct as Trait>::Type {} ], "<Struct as Trait>::Type {}");

    // PatKind::TupleStruct
    c1!(pat, [ Tuple() ], "Tuple()");
    c1!(pat, [ Tuple::<u8>() ], "Tuple::<u8>()");
    c1!(pat, [ Tuple::<'static>() ], "Tuple::<'static>()");
    c1!(pat, [ Tuple(x) ], "Tuple(x)");
    c1!(pat, [ Tuple(..) ], "Tuple(..)");
    c1!(pat, [ Tuple(x, ..) ], "Tuple(x, ..)");
    c1!(pat, [ <Struct as Trait>::Type() ], "<Struct as Trait>::Type()");

    // PatKind::Or
    c1!(pat, [ true | false ], "true | false");
    c2!(pat, [ | true ], "true", "| true");
    c2!(pat, [ |true| false ], "true | false", "|true| false");

    // PatKind::Path
    c1!(pat, [ crate::Path ], "crate::Path");
    c1!(pat, [ Path::<u8> ], "Path::<u8>");
    c1!(pat, [ Path::<'static> ], "Path::<'static>");
    c1!(pat, [ <Struct as Trait>::Type ], "<Struct as Trait>::Type");

    // PatKind::Tuple
    c1!(pat, [ () ], "()");
    c1!(pat, [ (true,) ], "(true,)");
    c1!(pat, [ (true, false) ], "(true, false)");

    // PatKind::Box
    c1!(pat, [ box pat ], "box pat");

    // PatKind::Ref
    c1!(pat, [ &pat ], "&pat");
    c1!(pat, [ &mut pat ], "&mut pat");

    // PatKind::Lit
    c1!(pat, [ 1_000_i8 ], "1_000_i8");

    // PatKind::Range
    c1!(pat, [ ..1 ], "..1");
    c1!(pat, [ 0.. ], "0..");
    c1!(pat, [ 0..1 ], "0..1");
    c1!(pat, [ 0..=1 ], "0..=1");
    c1!(pat, [ -2..=-1 ], "-2..=-1");

    // PatKind::Slice
    c1!(pat, [ [] ], "[]");
    c1!(pat, [ [true] ], "[true]");
    c2!(pat, [ [true,] ], "[true]", "[true,]");
    c1!(pat, [ [true, false] ], "[true, false]");

    // PatKind::Rest
    c1!(pat, [ .. ], "..");

    // PatKind::Never
    c1!(pat, [ Some(!) ], "Some(!)");
    c1!(pat, [ None | Some(!) ], "None | Some(!)");

    // PatKind::Paren
    c1!(pat, [ (pat) ], "(pat)");

    // PatKind::MacCall
    c1!(pat, [ mac!(...) ], "mac!(...)");
    c1!(pat, [ mac![...] ], "mac![...]");
    c1!(pat, [ mac! { ... } ], "mac! { ... }");
}

#[test]
fn test_path() {
    c1!(path, [ thing ], "thing");
    c1!(path, [ m::thing ], "m::thing");
    c1!(path, [ self::thing ], "self::thing");
    c1!(path, [ crate::thing ], "crate::thing");
    c1!(path, [ Self::thing ], "Self::thing");
    c1!(path, [ Self<'static> ], "Self<'static>");
    c2!(path, [ Self::<'static> ], "Self<'static>", "Self::<'static>");
    c1!(path, [ Self() ], "Self()");
    c1!(path, [ Self() -> () ], "Self() -> ()");
}

#[test]
fn test_stmt() {
    // StmtKind::Local
    c2!(stmt, [ let _ ], "let _;", "let _");
    c2!(stmt, [ let x = true ], "let x = true;", "let x = true");
    c2!(stmt, [ let x: bool = true ], "let x: bool = true;", "let x: bool = true");
    c2!(stmt, [ let (a, b) = (1, 2) ], "let (a, b) = (1, 2);", "let (a, b) = (1, 2)");
    c2!(stmt,
        [ let (a, b): (u32, u32) = (1, 2) ],
        "let (a, b): (u32, u32) = (1, 2);",
        "let (a, b): (u32, u32) = (1, 2)"
    );
    macro_rules! c2_let_expr_minus_one {
        ([ $expr:expr ], $stmt_expected:expr, $tokens_expected:expr $(,)?) => {
            c2!(stmt, [ let _ = $expr - 1 ], $stmt_expected, $tokens_expected);
        };
    }
    c2_let_expr_minus_one!(
        [ match void {} ],
        "let _ = match void {} - 1;",
        "let _ = match void {} - 1",
    );

    // StmtKind::Item
    c1!(stmt, [ struct S; ], "struct S;");
    c1!(stmt, [ struct S {} ], "struct S {}");

    // StmtKind::Expr
    c1!(stmt, [ loop {} ], "loop {}");

    // StmtKind::Semi
    c2!(stmt, [ 1 + 1 ], "1 + 1;", "1 + 1");
    macro_rules! c2_expr_as_stmt {
        // Parse as expr, then reparse as stmt.
        //
        // The c2_minus_one macro below can't directly call `c2!(stmt, ...)`
        // because `$expr - 1` cannot be parsed directly as a stmt. A statement
        // boundary occurs after the `match void {}`, after which the `-` token
        // hits "no rules expected this token in macro call".
        //
        // The unwanted statement boundary is exactly why the pretty-printer is
        // injecting parentheses around the subexpression, which is the behavior
        // we are interested in testing.
        ([ $expr:expr ], $stmt_expected:expr, $tokens_expected:expr $(,)?) => {
            c2!(stmt, [ $expr ], $stmt_expected, $tokens_expected);
        };
    }
    macro_rules! c2_minus_one {
        ([ $expr:expr ], $stmt_expected:expr, $tokens_expected:expr $(,)?) => {
            c2_expr_as_stmt!([ $expr - 1 ], $stmt_expected, $tokens_expected);
        };
    }
    c2_minus_one!(
        [ match void {} ],
        "(match void {}) - 1;",
        "match void {} - 1",
    );
    c2_minus_one!(
        [ match void {}() ],
        "(match void {})() - 1;",
        "match void {}() - 1",
    );
    c2_minus_one!(
        [ match void {}[0] ],
        "(match void {})[0] - 1;",
        "match void {}[0] - 1",
    );
    c2_minus_one!(
        [ loop { break 1; } ],
        "(loop { break 1; }) - 1;",
        "loop { break 1; } - 1",
    );

    // StmtKind::Empty
    c1!(stmt, [ ; ], ";");

    // StmtKind::MacCall
    c1!(stmt, [ mac!(...) ], "mac!(...)");
    c1!(stmt, [ mac![...] ], "mac![...]");
    c1!(stmt, [ mac! { ... } ], "mac! { ... }");
}

#[test]
fn test_ty() {
    // TyKind::Slice
    c1!(ty, [ [T] ], "[T]");

    // TyKind::Array
    c1!(ty, [ [T; 0] ], "[T; 0]");

    // TyKind::Ptr
    c1!(ty, [ *const T ], "*const T");
    c1!(ty, [ *mut T ], "*mut T");

    // TyKind::Ref
    c1!(ty, [ &T ], "&T");
    c1!(ty, [ &mut T ], "&mut T");
    c1!(ty, [ &'a T ], "&'a T");
    c1!(ty, [ &'a mut [T] ], "&'a mut [T]");
    c1!(ty, [ &A<B<C<D<E>>>> ], "&A<B<C<D<E>>>>");
    c2!(ty, [ &A<B<C<D<E> > > > ], "&A<B<C<D<E>>>>", "&A<B<C<D<E> > > >");

    // TyKind::BareFn
    c1!(ty, [ fn() ], "fn()");
    c1!(ty, [ fn() -> () ], "fn() -> ()");
    c1!(ty, [ fn(u8) ], "fn(u8)");
    c1!(ty, [ fn(x: u8) ], "fn(x: u8)");
    c2!(ty, [ for<> fn() ], "fn()", "for<> fn()");
    c1!(ty, [ for<'a> fn() ], "for<'a> fn()");

    // TyKind::Never
    c1!(ty, [ ! ], "!");

    // TyKind::Tup
    c1!(ty, [ () ], "()");
    c1!(ty, [ (T,) ], "(T,)");
    c1!(ty, [ (T, U) ], "(T, U)");

    // TyKind::AnonStruct: untestable in isolation.

    // TyKind::AnonUnion: untestable in isolation.

    // TyKind::Path
    c1!(ty, [ T ], "T");
    c1!(ty, [ Ref<'a> ], "Ref<'a>");
    c1!(ty, [ PhantomData<T> ], "PhantomData<T>");
    c2!(ty, [ PhantomData::<T> ], "PhantomData<T>", "PhantomData::<T>");
    c1!(ty, [ Fn() -> ! ], "Fn() -> !");
    c1!(ty, [ Fn(u8) -> ! ], "Fn(u8) -> !");
    c1!(ty, [ <Struct as Trait>::Type ], "<Struct as Trait>::Type");

    // TyKind::TraitObject
    c1!(ty, [ dyn Send ], "dyn Send");
    c1!(ty, [ dyn Send + 'a ], "dyn Send + 'a");
    c1!(ty, [ dyn 'a + Send ], "dyn 'a + Send");
    c1!(ty, [ dyn ?Sized ], "dyn ?Sized");
    c1!(ty, [ dyn ~const Clone ], "dyn ~const Clone");
    c1!(ty, [ dyn for<'a> Send ], "dyn for<'a> Send");

    // TyKind::ImplTrait
    c1!(ty, [ impl Send ], "impl Send");
    c1!(ty, [ impl Send + 'a ], "impl Send + 'a");
    c1!(ty, [ impl 'a + Send ], "impl 'a + Send");
    c1!(ty, [ impl ?Sized ], "impl ?Sized");
    c1!(ty, [ impl ~const Clone ], "impl ~const Clone");
    c1!(ty, [ impl for<'a> Send ], "impl for<'a> Send");

    // TyKind::Paren
    c1!(ty, [ (T) ], "(T)");

    // TyKind::Typeof: unused for now.

    // TyKind::Infer
    c1!(ty, [ _ ], "_");

    // TyKind::ImplicitSelf: there is no syntax for this.

    // TyKind::MacCall
    c1!(ty, [ mac!(...) ], "mac!(...)");
    c1!(ty, [ mac![...] ], "mac![...]");
    c1!(ty, [ mac! { ... } ], "mac! { ... }");

    // TyKind::Err: untestable.

    // TyKind::CVarArgs
    // FIXME: todo
}

#[test]
fn test_vis() {
    // VisibilityKind::Public
    c2!(vis, [ pub ], "pub ", "pub");

    // VisibilityKind::Restricted
    c2!(vis, [ pub(crate) ], "pub(crate) ", "pub(crate)");
    c2!(vis, [ pub(self) ], "pub(self) ", "pub(self)");
    c2!(vis, [ pub(super) ], "pub(super) ", "pub(super)");
    c2!(vis, [ pub(in crate) ], "pub(in crate) ", "pub(in crate)");
    c2!(vis, [ pub(in self) ], "pub(in self) ", "pub(in self)");
    c2!(vis, [ pub(in super) ], "pub(in super) ", "pub(in super)");
    c2!(vis, [ pub(in path::to) ], "pub(in path::to) ", "pub(in path::to)");
    c2!(vis, [ pub(in ::path::to) ], "pub(in ::path::to) ", "pub(in ::path::to)");
    c2!(vis, [ pub(in self::path::to) ], "pub(in self::path::to) ", "pub(in self::path::to)");
    c2!(vis,
        [ pub(in super::path::to) ],
        "pub(in super::path::to) ",
        "pub(in super::path::to)"
    );

    // VisibilityKind::Inherited
    // This one is different because directly calling `vis!` does not work.
    macro_rules! inherited_vis { ($vis:vis struct) => { vis!($vis) }; }
    assert_eq!(inherited_vis!(struct), "");
    assert_eq!(stringify!(), "");
}

macro_rules! p {
    ([$($tt:tt)*], $s:literal) => {
        assert_eq!(stringify!($($tt)*), $s);
    };
}

#[test]
fn test_punct() {
    // For all these cases, we should preserve spaces between the tokens.
    // Otherwise, any old proc macro that parses pretty-printed code might glue
    // together tokens that shouldn't be glued.
    p!([ = = < < <= <= == == != != >= >= > > ], "= = < < <= <= == == != != >= >= > >");
    p!([ && && & & || || | | ! ! ], "&& && & & || || | | ! !");
    p!([ ~ ~ @ @ # # ], "~ ~ @ @ # #");
    p!([ . . .. .. ... ... ..= ..=], ". . .. .. ... ... ..= ..=");
    p!([ , , ; ; : : :: :: ], ", , ; ; : : :: ::");
    p!([ -> -> <- <- => =>], "-> -> <- <- => =>");
    p!([ $ $ ? ? ' ' ], "$ $ ? ? ' '");
    p!([ + + += += - - -= -= * * *= *= / / /= /= ], "+ + += += - - -= -= * * *= *= / / /= /=");
    p!([ % % %= %= ^ ^ ^= ^= << << <<= <<= >> >> >>= >>= ],
        "% % %= %= ^ ^ ^= ^= << << <<= <<= >> >> >>= >>=");
    p!([ +! ?= |> >>@ --> <-- $$ =====> ], "+! ?= |> >>@ --> <-- $$ =====>");
    p!([ ,; ;, ** @@ $+$ >< <> ?? +== ], ",; ;, ** @@ $+$ >< <> ?? +==");
    p!([ :#!@|$=&*,+;*~? ], ":#!@|$=&*,+;*~?");
}
