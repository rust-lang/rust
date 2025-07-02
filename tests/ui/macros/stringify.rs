//@ run-pass
//@ edition:2024
//@ compile-flags: --test

#![allow(incomplete_features)]
#![feature(auto_traits)]
#![feature(box_patterns)]
#![feature(const_trait_impl)]
#![feature(coroutines)]
#![feature(decl_macro)]
#![feature(explicit_tail_calls)]
#![feature(if_let_guard)]
#![feature(more_qualified_paths)]
#![feature(never_patterns)]
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

macro_rules! c1 {
    ($frag:ident, [$($tt:tt)*], $s:literal) => {
        // Prior to #125174:
        // - the first of these two lines created a `TokenKind::Interpolated`
        //   that was printed by the AST pretty printer;
        // - the second of these two lines created a token stream that was
        //   printed by the TokenStream pretty printer.
        //
        // Now they are both printed by the TokenStream pretty printer. But it
        // doesn't hurt to keep both assertions to ensure this remains true.
        //
        // (This also explains the name `c1`. There used to be a `c2` macro for
        // cases where the two pretty printers produced different output.)
        assert_eq!($frag!($($tt)*), $s);
        assert_eq!(stringify!($($tt)*), $s);
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

    // Attributes are not allowed on vanilla blocks.
}

#[test]
fn test_expr() {
    // ExprKind::Array
    c1!(expr, [ [] ], "[]");
    c1!(expr, [ [true] ], "[true]");
    c1!(expr, [ [true,] ], "[true,]");
    c1!(expr, [ [true, true] ], "[true, true]");

    // ExprKind::ConstBlock
    // FIXME: todo

    // ExprKind::Call
    c1!(expr, [ f() ], "f()");
    c1!(expr, [ f::<u8>() ], "f::<u8>()");
    c1!(expr, [ f ::  < u8>( ) ], "f :: < u8>()");
    c1!(expr, [ f::<1>() ], "f::<1>()");
    c1!(expr, [ f::<'a, u8, 1>() ], "f::<'a, u8, 1>()");
    c1!(expr, [ f(true) ], "f(true)");
    c1!(expr, [ f(true,) ], "f(true,)");
    c1!(expr, [ ()() ], "()()");

    // ExprKind::MethodCall
    c1!(expr, [ x.f() ], "x.f()");
    c1!(expr, [ x.f::<u8>() ], "x.f::<u8>()");
    c1!(expr, [ x.collect::<Vec<_>>() ], "x.collect::<Vec<_>>()");

    // ExprKind::Tup
    c1!(expr, [ () ], "()");
    c1!(expr, [ (true,) ], "(true,)");
    c1!(expr, [ (true, false) ], "(true, false)");
    c1!(expr, [ (true, false,) ], "(true, false,)");

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
    c1!(expr, [ 1 + || {} ], "1 + || {}");

    // ExprKind::Block
    c1!(expr, [ {} ], "{}");
    c1!(expr, [ unsafe {} ], "unsafe {}");
    c1!(expr, [ 'a: {} ], "'a: {}");
    c1!(expr, [ #[attr] {} ], "#[attr] {}");
    c1!(expr,
        [
            {
                #![attr]
            }
        ],
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
    c1!(expr, [ lo .. hi ], "lo .. hi");
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
    c1!(expr, [ mac!() ], "mac!()");
    c1!(expr, [ mac![] ], "mac![]");
    c1!(expr, [ mac! {} ], "mac! {}");
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

    // Ones involving attributes.
    c1!(expr, [ #[aa] 1 ], "#[aa] 1");
    c1!(expr, [ #[aa] #[bb] x ], "#[aa] #[bb] x");
    c1!(expr, [ #[aa] 1 + 2 ], "#[aa] 1 + 2");
    c1!(expr, [ #[aa] x + 2 ], "#[aa] x + 2");
    c1!(expr, [ #[aa] 1 / #[bb] 2 ], "#[aa] 1 / #[bb] 2");
    c1!(expr, [ #[aa] x / #[bb] 2 ], "#[aa] x / #[bb] 2");
    c1!(expr, [ 1 << #[bb] 2 ], "1 << #[bb] 2");
    c1!(expr, [ x << #[bb] 2 ], "x << #[bb] 2");
    c1!(expr, [ #[aa] (1 + 2) ], "#[aa] (1 + 2)");
    c1!(expr, [ #[aa] #[bb] (x + 2) ], "#[aa] #[bb] (x + 2)");
    c1!(expr, [ #[aa] x[0].p ], "#[aa] x[0].p");
    c1!(expr, [ #[aa] { #![bb] 0 } ], "#[aa] { #![bb] 0 }");
}

#[test]
fn test_item() {
    // ItemKind::ExternCrate
    c1!(item, [ extern crate std; ], "extern crate std;");
    c1!(item, [ pub extern crate self as std; ], "pub extern crate self as std;");

    // ItemKind::Use
    c1!(item, [ pub use crate::{a, b::c}; ], "pub use crate::{a, b::c};");
    c1!(item, [ pub use crate::{ e, ff }; ], "pub use crate::{ e, ff };");
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
    c1!(item, [ pub extern "C" {} ], "pub extern \"C\" {}");
    c1!(item, [ unsafe extern "C++" {} ], "unsafe extern \"C++\" {}");

    // ItemKind::GlobalAsm: untestable because this test works pre-expansion.

    // ItemKind::TyAlias
    c1!(item,
        [
            pub default type Type<'a>: Bound
            where
                Self: 'a,
            = T;
        ],
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
    c1!(item,
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
        "enum Enum<T> where T: 'a, { Unit, Tuple(T), Struct { t: T }, }"
    );

    // ItemKind::Struct
    c1!(item, [ pub struct Unit; ], "pub struct Unit;");
    c1!(item, [ struct Tuple(); ], "struct Tuple();");
    c1!(item, [ struct Tuple(T); ], "struct Tuple(T);");
    c1!(item, [ struct Struct {} ], "struct Struct {}");
    c1!(item,
        [
            struct Struct<T>
            where
                T: 'a,
            {
                t: T,
            }
        ],
        "struct Struct<T> where T: 'a, { t: T, }"
    );

    // ItemKind::Union
    c1!(item, [ pub union Union {} ], "pub union Union {}");
    c1!(item,
        [
            union Union<T> where T: 'a {
                t: T,
            }
        ],
        "union Union<T> where T: 'a { t: T, }"
    );

    // ItemKind::Trait
    c1!(item, [ pub unsafe auto trait Send {} ], "pub unsafe auto trait Send {}");
    c1!(item,
        [
            trait Trait<'a>: Sized
            where
                Self: 'a,
            {
            }
        ],
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

    // ItemKind::MacCall
    c1!(item, [ mac!(); ], "mac!();");
    c1!(item, [ mac![]; ], "mac![];");
    c1!(item, [ mac! {} ], "mac! {}");
    c1!(item, [ mac!(...); ], "mac!(...);");
    c1!(item, [ mac![...]; ], "mac![...];");
    c1!(item, [ mac! {...} ], "mac! {...}");

    // ItemKind::MacroDef
    c1!(item,
        [
            macro_rules! stringify {
                () => {};
            }
        ],
        "macro_rules! stringify { () => {}; }"
    );
    c1!(item, [ pub macro stringify() {} ], "pub macro stringify() {}");

    // Ones involving attributes.
    c1!(item, [ #[aa] mod m; ], "#[aa] mod m;");
    c1!(item, [ mod m { #![bb] } ], "mod m { #![bb] }");
    c1!(item, [ #[aa] mod m { #![bb] } ], "#[aa] mod m { #![bb] }");
}

#[test]
fn test_meta() {
    c1!(meta, [ k ], "k");
    c1!(meta, [ k = "v" ], "k = \"v\"");
    c1!(meta, [ list(k1, k2 = "v") ], "list(k1, k2 = \"v\")");
    c1!(meta, [ serde::k ], "serde::k");

    // Attributes are not allowed on metas.
}

#[test]
fn test_pat() {
    // PatKind::Missing: untestable in isolation.

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
    c1!(pat, [ Struct ::< u8 > {} ], "Struct ::< u8 > {}");
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
    c1!(pat, [ | true ], "| true");
    c1!(pat, [ |true| false ], "|true| false");

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

    // PatKind::Expr
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
    c1!(pat, [ [true,] ], "[true,]");
    c1!(pat, [ [true, false] ], "[true, false]");

    // PatKind::Rest
    c1!(pat, [ .. ], "..");

    // PatKind::Never
    c1!(pat, [ Some(!) ], "Some(!)");
    c1!(pat, [ None | Some(!) ], "None | Some(!)");

    // PatKind::Paren
    c1!(pat, [ (pat) ], "(pat)");

    // PatKind::MacCall
    c1!(pat, [ mac!() ], "mac!()");
    c1!(pat, [ mac![] ], "mac![]");
    c1!(pat, [ mac! {} ], "mac! {}");
    c1!(pat, [ mac!(...) ], "mac!(...)");
    c1!(pat, [ mac! [ ... ] ], "mac! [...]");
    c1!(pat, [ mac! { ... } ], "mac! { ... }");

    // Attributes are not allowed on patterns.
}

#[test]
fn test_path() {
    c1!(path, [ thing ], "thing");
    c1!(path, [ m::thing ], "m::thing");
    c1!(path, [ self::thing ], "self::thing");
    c1!(path, [ crate::thing ], "crate::thing");
    c1!(path, [ Self::thing ], "Self::thing");
    c1!(path, [ Self<'static> ], "Self<'static>");
    c1!(path, [ Self::<'static> ], "Self::<'static>");
    c1!(path, [ Self() ], "Self()");
    c1!(path, [ Self() -> () ], "Self() -> ()");

    // Attributes are not allowed on paths.
}

#[test]
fn test_stmt() {
    // StmtKind::Local
    c1!(stmt, [ let _ ], "let _");
    c1!(stmt, [ let x = true ], "let x = true");
    c1!(stmt, [ let x: bool = true ], "let x: bool = true");
    c1!(stmt, [ let (a, b) = (1, 2) ], "let (a, b) = (1, 2)");
    c1!(stmt, [ let (a, b): (u32, u32) = (1, 2) ], "let (a, b): (u32, u32) = (1, 2)");
    c1!(stmt, [ let _ = f() else { return; } ], "let _ = f() else { return; }");

    // StmtKind::Item
    c1!(stmt, [ struct S; ], "struct S;");
    c1!(stmt, [ struct S {} ], "struct S {}");

    // StmtKind::Expr
    c1!(stmt, [ loop {} ], "loop {}");

    // StmtKind::Semi
    c1!(stmt, [ 1 + 1 ], "1 + 1");

    // StmtKind::Empty
    c1!(stmt, [ ; ], ";");

    // StmtKind::MacCall
    c1!(stmt, [ mac! ( ) ], "mac! ()");
    c1!(stmt, [ mac![] ], "mac![]");
    c1!(stmt, [ mac!{} ], "mac!{}");
    c1!(stmt, [ mac!(...) ], "mac!(...)");
    c1!(stmt, [ mac![...] ], "mac![...]");
    c1!(stmt, [ mac! { ... } ], "mac! { ... }");

    // Ones involving attributes.
    c1!(stmt, [ #[aa] 1 ], "#[aa] 1");
    c1!(stmt, [ #[aa] #[bb] x ], "#[aa] #[bb] x");
    c1!(stmt, [ #[aa] 1 as u32 ], "#[aa] 1 as u32");
    c1!(stmt, [ #[aa] x as u32 ], "#[aa] x as u32");
    c1!(stmt, [ #[aa] 1 .. #[bb] 2 ], "#[aa] 1 .. #[bb] 2");
    c1!(stmt, [ #[aa] x .. #[bb] 2 ], "#[aa] x .. #[bb] 2");
    c1!(stmt, [ 1 || #[bb] 2 ], "1 || #[bb] 2");
    c1!(stmt, [ x || #[bb] 2 ], "x || #[bb] 2");
    c1!(stmt, [ #[aa] (1 + 2) ], "#[aa] (1 + 2)");
    c1!(stmt, [ #[aa] #[bb] (x + 2) ], "#[aa] #[bb] (x + 2)");
    c1!(stmt, [ #[aa] x[0].p ], "#[aa] x[0].p");
    c1!(stmt, [ #[aa] { #![bb] 0 } ], "#[aa] { #![bb] 0 }");
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
    c1!(ty, [ &A<B<C<D<E> > > > ], "&A<B<C<D<E> > > >");

    // TyKind::BareFn
    c1!(ty, [ fn() ], "fn()");
    c1!(ty, [ fn() -> () ], "fn() -> ()");
    c1!(ty, [ fn(u8) ], "fn(u8)");
    c1!(ty, [ fn(x: u8) ], "fn(x: u8)");
    c1!(ty, [ for<> fn() ], "for<> fn()");
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
    c1!(ty, [ PhantomData::<T> ], "PhantomData::<T>");
    c1!(ty, [ Fn() -> ! ], "Fn() -> !");
    c1!(ty, [ Fn(u8) -> ! ], "Fn(u8) -> !");
    c1!(ty, [ <Struct as Trait>::Type ], "<Struct as Trait>::Type");

    // TyKind::TraitObject
    c1!(ty, [ dyn Send ], "dyn Send");
    c1!(ty, [ dyn Send + 'a ], "dyn Send + 'a");
    c1!(ty, [ dyn 'a + Send ], "dyn 'a + Send");
    c1!(ty, [ dyn ?Sized ], "dyn ?Sized");
    c1!(ty, [ dyn [const] Clone ], "dyn [const] Clone");
    c1!(ty, [ dyn for<'a> Send ], "dyn for<'a> Send");

    // TyKind::ImplTrait
    c1!(ty, [ impl Send ], "impl Send");
    c1!(ty, [ impl Send + 'a ], "impl Send + 'a");
    c1!(ty, [ impl 'a + Send ], "impl 'a + Send");
    c1!(ty, [ impl ?Sized ], "impl ?Sized");
    c1!(ty, [ impl [const] Clone ], "impl [const] Clone");
    c1!(ty, [ impl for<'a> Send ], "impl for<'a> Send");

    // TyKind::Paren
    c1!(ty, [ (T) ], "(T)");

    // TyKind::Typeof: unused for now.

    // TyKind::Infer
    c1!(ty, [ _ ], "_");

    // TyKind::ImplicitSelf: there is no syntax for this.

    // TyKind::MacCall
    c1!(ty, [ mac!() ], "mac!()");
    c1!(ty, [ mac![] ], "mac![]");
    c1!(ty, [ mac! { } ], "mac! {}");
    c1!(ty, [ mac!(...) ], "mac!(...)");
    c1!(ty, [ mac![...] ], "mac![...]");
    c1!(ty, [ mac! { ... } ], "mac! { ... }");

    // TyKind::Err: untestable.

    // TyKind::CVarArgs
    // FIXME: todo

    // Attributes are not allowed on types.
}

#[test]
fn test_vis() {
    // VisibilityKind::Public
    c1!(vis, [ pub ], "pub");

    // VisibilityKind::Restricted
    c1!(vis, [ pub(crate) ], "pub(crate)");
    c1!(vis, [ pub(self) ], "pub(self)");
    c1!(vis, [ pub(super) ], "pub(super)");
    c1!(vis, [ pub(in crate) ], "pub(in crate)");
    c1!(vis, [ pub(in self) ], "pub(in self)");
    c1!(vis, [ pub(in super) ], "pub(in super)");
    c1!(vis, [ pub(in path::to) ], "pub(in path::to)");
    c1!(vis, [ pub(in ::path::to) ], "pub(in ::path::to)");
    c1!(vis, [ pub(in self::path::to) ], "pub(in self::path::to)");
    c1!(vis, [ pub(in super::path::to) ], "pub(in super::path::to)");

    // VisibilityKind::Inherited
    // This one is different because directly calling `vis!` does not work.
    macro_rules! inherited_vis { ($vis:vis struct) => { vis!($vis) }; }
    assert_eq!(inherited_vis!(struct), "");
    assert_eq!(stringify!(), "");

    // Attributes are not allowed on visibilities.
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
