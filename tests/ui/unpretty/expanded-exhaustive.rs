//@ compile-flags: -Zunpretty=expanded
//@ edition:2024
//@ check-pass

#![feature(auto_traits)]
#![feature(box_patterns)]
#![feature(builtin_syntax)]
#![feature(concat_idents)]
#![feature(const_trait_impl)]
#![feature(decl_macro)]
#![feature(deref_patterns)]
#![feature(explicit_tail_calls)]
#![feature(gen_blocks)]
#![feature(let_chains)]
#![feature(more_qualified_paths)]
#![feature(never_patterns)]
#![feature(never_type)]
#![feature(pattern_types)]
#![feature(pattern_type_macro)]
#![feature(prelude_import)]
#![feature(specialization)]
#![feature(trace_macros)]
#![feature(trait_alias)]
#![feature(try_blocks)]
#![feature(yeet_expr)]
#![allow(incomplete_features)]

#[prelude_import]
use self::prelude::*;

mod prelude {
    pub use std::prelude::rust_2024::*;

    pub type T = _;

    pub trait Trait {
        const CONST: ();
    }
}

mod attributes {
    //! inner single-line doc comment
    /*!
     * inner multi-line doc comment
     */
    #![doc = "inner doc attribute"]
    #![allow(dead_code, unused_variables)]
    #![no_std]

    /// outer single-line doc comment
    /**
     * outer multi-line doc comment
     */
    #[doc = "outer doc attribute"]
    #[doc = concat!("mac", "ro")]
    #[allow()]
    #[repr(C)]
    struct Struct;
}

mod expressions {
    /// ExprKind::Array
    fn expr_array() {
        [];
        [true];
        [true,];
        [true, true];
        ["long........................................................................"];
        ["long............................................................", true];
    }

    /// ExprKind::ConstBlock
    fn expr_const_block() {
        const {};
        const { 1 };
        const { struct S; };
    }

    /// ExprKind::Call
    fn expr_call() {
        let f;
        f();
        f::<u8>();
        f::<1>();
        f::<'static, u8, 1>();
        f(true);
        f(true,);
        ()();
    }

    /// ExprKind::MethodCall
    fn expr_method_call() {
        let x;
        x.f();
        x.f::<u8>();
        x.collect::<Vec<_>>();
    }

    /// ExprKind::Tup
    fn expr_tup() {
        ();
        (true,);
        (true, false);
        (true, false,);
    }

    /// ExprKind::Binary
    fn expr_binary() {
        let (a, b, c, d, x, y);
        true || false;
        true || false && false;
        a < 1 && 2 < b && c > 3 && 4 > d;
        a & b & !c;
        a + b * c - d + -1 * -2 - -3;
        x = !y;
    }

    /// ExprKind::Unary
    fn expr_unary() {
        let expr;
        *expr;
        !expr;
        -expr;
    }

    /// ExprKind::Lit
    fn expr_lit() {
        'x';
        1_000_i8;
        1.00000000000000000000001;
    }

    /// ExprKind::Cast
    fn expr_cast() {
        let expr;
        expr as T;
        expr as T<u8>;
    }

    /// ExprKind::Type
    fn expr_type() {
        let expr;
        builtin # type_ascribe(expr, T);
    }

    /// ExprKind::Let
    fn expr_let() {
        let b;
        if let Some(a) = b {}
        if let _ = true && false {}
        if let _ = (true && false) {}
    }

    /// ExprKind::If
    fn expr_if() {
        if true {}
        if !true {}
        if let true = true {} else {}
        if true {} else if false {}
        if true {} else if false {} else {}
        if true { return; } else if false { 0 } else { 0 }
    }

    /// ExprKind::While
    fn expr_while() {
        while false {}
        'a: while false {}
        while let true = true {}
    }

    /// ExprKind::ForLoop
    fn expr_for_loop() {
        let x;
        for _ in x {}
        'a: for _ in x {}
    }

    /// ExprKind::Loop
    fn expr_loop() {
        loop {}
        'a: loop {}
    }

    /// ExprKind::Match
    fn expr_match() {
        let value;
        match value {}
        match value { ok => 1 }
        match value {
            ok => 1,
            err => 0,
        }
    }

    /// ExprKind::Closure
    fn expr_closure() {
        let value;
        || {};
        |x| {};
        |x: u8| {};
        || ();
        move || value;
        async || value;
        async move || value;
        static || value;
        static move || value;
        (static async || value);
        (static async move || value);
        || -> u8 { value };
        1 + || {};
    }

    /// ExprKind::Block
    fn expr_block() {
        {}
        unsafe {}
        'a: {}
        #[allow()] {}
        { #![allow()] }
    }

    /// ExprKind::Gen
    fn expr_gen() {
        async {};
        async move {};
        gen {};
        gen move {};
        async gen {};
        async gen move {};
    }

    /// ExprKind::Await
    fn expr_await() {
        let fut;
        fut.await;
    }

    /// ExprKind::TryBlock
    fn expr_try_block() {
        try {}
        try { return; }
    }

    /// ExprKind::Assign
    fn expr_assign() {
        let expr;
        expr = true;
    }

    /// ExprKind::AssignOp
    fn expr_assign_op() {
        let expr;
        expr += true;
    }

    /// ExprKind::Field
    fn expr_field() {
        let expr;
        expr.field;
        expr.0;
    }

    /// ExprKind::Index
    fn expr_index() {
        let expr;
        expr[true];
    }

    /// ExprKind::Range
    fn expr_range() {
        let (lo, hi);
        ..;
        ..hi;
        lo..;
        lo..hi;
        lo .. hi;
        ..=hi;
        lo..=hi;
        -2..=-1;
    }

    /// ExprKind::Underscore
    fn expr_underscore() {
        _;
    }

    /// ExprKind::Path
    fn expr_path() {
        let x;
        crate::expressions::expr_path;
        crate::expressions::expr_path::<'static>;
        <T as Default>::default;
        <T as ::core::default::Default>::default::<>;
        x::();
        x::(T, T) -> T;
        crate::() -> ()::expressions::() -> ()::expr_path;
        core::()::marker::()::PhantomData;
    }

    /// ExprKind::AddrOf
    fn expr_addr_of() {
        let expr;
        &expr;
        &mut expr;
        &raw const expr;
        &raw mut expr;
    }

    /// ExprKind::Break
    fn expr_break() {
        'a: {
            break;
            break 'a;
            break true;
            break 'a true;
        }
    }

    /// ExprKind::Continue
    fn expr_continue() {
        'a: {
            continue;
            continue 'a;
        }
    }

    /// ExprKind::Ret
    fn expr_ret() {
        return;
        return true;
    }

    /// ExprKind::InlineAsm
    fn expr_inline_asm() {
        let x;
        core::arch::asm!(
            "mov {tmp}, {x}",
            "shl {tmp}, 1",
            "shl {x}, 2",
            "add {x}, {tmp}",
            x = inout(reg) x,
            tmp = out(reg) _,
        );
    }

    /// ExprKind::OffsetOf
    fn expr_offset_of() {
        core::mem::offset_of!(T, field);
    }

    /// ExprKind::MacCall
    fn expr_mac_call() {
        stringify!(...);
        stringify![...];
        stringify! { ... };
    }

    /// ExprKind::Struct
    fn expr_struct() {
        struct Struct {}
        let (x, base);
        Struct {};
        <Struct as ToOwned>::Owned {};
        Struct { .. };
        Struct { .. base };
        Struct { x };
        Struct { x, ..base };
        Struct { x: true };
        Struct { x: true, .. };
        Struct { x: true, ..base };
        Struct { 0: true, ..base };
    }

    /// ExprKind::Repeat
    fn expr_repeat() {
        [(); 0];
    }

    /// ExprKind::Paren
    fn expr_paren() {
        let expr;
        (expr);
    }

    /// ExprKind::Try
    fn expr_try() {
        let expr;
        expr?;
    }

    /// ExprKind::Yield
    fn expr_yield() {
        yield;
        yield true;
    }

    /// ExprKind::Yeet
    fn expr_yeet() {
        do yeet;
        do yeet 0;
    }

    /// ExprKind::Become
    fn expr_become() {
        become true;
    }

    /// ExprKind::IncludedBytes
    fn expr_include_bytes() {
        include_bytes!("auxiliary/data.txt");
    }

    /// ExprKind::FormatArgs
    fn expr_format_args() {
        let expr;
        format_args!("");
        format_args!("{}", expr);
    }
}

mod items {
    /// ItemKind::ExternCrate
    mod item_extern_crate {
        extern crate core;
        extern crate self as unpretty;
        pub extern crate core as _;
    }

    /// ItemKind::Use
    mod item_use {
        use crate::{expressions, items::item_use};
        pub use core::*;
    }

    /// ItemKind::Static
    mod item_static {
        pub static A: () = {};
        static mut B: () = {};
    }

    /// ItemKind::Const
    mod item_const {
        pub const A: () = {};

        trait TraitItems {
            const B: ();
            const C: () = {};
        }
    }

    /// ItemKind::Fn
    mod item_fn {
        pub const unsafe extern "C" fn f() {}
        pub async unsafe extern fn g() {}
        fn h<'a, T>() where T: 'a {}

        trait TraitItems {
            unsafe extern fn f();
        }

        impl TraitItems for _ {
            default unsafe extern fn f() {}
        }
    }

    /// ItemKind::Mod
    mod item_mod {
        // ...
    }

    /// ItemKind::ForeignMod
    mod item_foreign_mod {
        unsafe extern "C++" {}
        unsafe extern {}
    }

    /// ItemKind::GlobalAsm
    mod item_global_asm {
        core::arch::global_asm!(".globl my_asm_func");
    }

    /// ItemKind::TyAlias
    mod item_ty_alias {
        pub type Type<'a> where T: 'a, = T;
    }

    /// ItemKind::Enum
    mod item_enum {
        pub enum Void {}
        enum Empty {
            Unit,
            Tuple(),
            Struct {},
        }
        enum Generic<'a, T>
        where
            T: 'a,
        {
            Tuple(T),
            Struct { t: T },
        }
    }

    /// ItemKind::Struct
    mod item_struct {
        pub struct Unit;
        struct Tuple();
        struct Newtype(Unit);
        struct Struct {}
        struct Generic<'a, T>
        where
            T: 'a,
        {
            t: T,
        }
    }

    /// ItemKind::Union
    mod item_union {
        union Generic<'a, T>
        where
            T: 'a,
        {
            t: T,
        }
    }

    /// ItemKind::Trait
    mod item_trait {
        pub unsafe auto trait Send {}
        trait Trait<'a>: Sized
        where
            Self: 'a,
        {
        }
    }

    /// ItemKind::TraitAlias
    mod item_trait_alias {
        pub trait Trait<T> = Sized where for<'a> T: 'a;
    }

    /// ItemKind::Impl
    mod item_impl {
        impl () {}
        impl<T> () {}
        impl Default for () {}
        impl<T> const Default for () {}
    }

    /// ItemKind::MacCall
    mod item_mac_call {
        trace_macros!(false);
        trace_macros![false];
        trace_macros! { false }
    }

    /// ItemKind::MacroDef
    mod item_macro_def {
        macro_rules! mac { () => {...}; }
        pub macro stringify() {}
    }

    /// ItemKind::Delegation
    mod item_delegation {
        /*! FIXME: todo */
    }

    /// ItemKind::DelegationMac
    mod item_delegation_mac {
        /*! FIXME: todo */
    }
}

mod patterns {
    /// PatKind::Wild
    fn pat_wild() {
        let _;
    }

    /// PatKind::Ident
    fn pat_ident() {
        let x;
        let ref x;
        let mut x;
        let ref mut x;
        let ref mut x @ _;
    }

    /// PatKind::Struct
    fn pat_struct() {
        let T {};
        let T::<T> {};
        let T::<'static> {};
        let T { x };
        let T { x: _x };
        let T { .. };
        let T { x, .. };
        let T { x: _x, .. };
        let T { 0: _x, .. };
        let <T as ToOwned>::Owned {};
    }

    /// PatKind::TupleStruct
    fn pat_tuple_struct() {
        struct Tuple();
        let Tuple();
        let Tuple::<T>();
        let Tuple::<'static>();
        let Tuple(x);
        let Tuple(..);
        let Tuple(x, ..);
    }

    /// PatKind::Or
    fn pat_or() {
        let (true | false);
        let (| true);
        let (|true| false);
    }

    /// PatKind::Path
    fn pat_path() {
        let core::marker::PhantomData;
        let core::marker::PhantomData::<T>;
        let core::marker::PhantomData::<'static>;
        let <T as Trait>::CONST;
    }

    /// PatKind::Tuple
    fn pat_tuple() {
        let ();
        let (true,);
        let (true, false);
    }

    /// PatKind::Box
    fn pat_box() {
        let box pat;
    }

    /// PatKind::Deref
    fn pat_deref() {
        let deref!(pat);
    }

    /// PatKind::Ref
    fn pat_ref() {
        let &pat;
        let &mut pat;
    }

    /// PatKind::Lit
    fn pat_lit() {
        let 1_000_i8;
        let -"";
    }

    /// PatKind::Range
    fn pat_range() {
        let ..1;
        let 0..;
        let 0..1;
        let 0..=1;
        let -2..=-1;
    }

    /// PatKind::Slice
    fn pat_slice() {
        let [];
        let [true];
        let [true,];
        let [true, false];
    }

    /// PatKind::Rest
    fn pat_rest() {
        let ..;
    }

    /// PatKind::Never
    fn pat_never() {
        let !;
        let Some(!);
    }

    /// PatKind::Paren
    fn pat_paren() {
        let (pat);
    }

    /// PatKind::MacCall
    fn pat_mac_call() {
        let stringify!();
        let stringify![];
        let stringify! {};
    }
}

mod statements {
    /// StmtKind::Let
    fn stmt_let() {
        let _;
        let _ = true;
        let _: T = true;
        let _ = true else { return; };
    }

    /// StmtKind::Item
    fn stmt_item() {
        struct Struct {}
        struct Unit;
    }

    /// StmtKind::Expr
    fn stmt_expr() {
        ()
    }

    /// StmtKind::Semi
    fn stmt_semi() {
        1 + 1;
    }

    /// StmtKind::Empty
    fn stmt_empty() {
        ;
    }

    /// StmtKind::MacCall
    fn stmt_mac_call() {
        stringify!(...);
        stringify![...];
        stringify! { ... };
    }
}

mod types {
    /// TyKind::Slice
    fn ty_slice() {
        let _: [T];
    }

    /// TyKind::Array
    fn ty_array() {
        let _: [T; 0];
    }

    /// TyKind::Ptr
    fn ty_ptr() {
        let _: *const T;
        let _: *mut T;
    }

    /// TyKind::Ref
    fn ty_ref() {
        let _: &T;
        let _: &mut T;
        let _: &'static T;
        let _: &'static mut [T];
        let _: &T<T<T<T<T>>>>;
        let _: &T<T<T<T<T> > > >;
    }

    /// TyKind::BareFn
    fn ty_bare_fn() {
        let _: fn();
        let _: fn() -> ();
        let _: fn(T);
        let _: fn(t: T);
        let _: for<> fn();
        let _: for<'a> fn();
    }

    /// TyKind::Never
    fn ty_never() {
        let _: !;
    }

    /// TyKind::Tup
    fn ty_tup() {
        let _: ();
        let _: (T,);
        let _: (T, T);
    }

    /// TyKind::Path
    fn ty_path() {
        let _: T;
        let _: T<'static>;
        let _: T<T>;
        let _: T::<T>;
        let _: T() -> !;
        let _: <T as ToOwned>::Owned;
    }

    /// TyKind::TraitObject
    fn ty_trait_object() {
        let _: dyn Send;
        let _: dyn Send + 'static;
        let _: dyn 'static + Send;
        let _: dyn for<'a> Send;
    }

    /// TyKind::ImplTrait
    const fn ty_impl_trait() {
        let _: impl Send;
        let _: impl Send + 'static;
        let _: impl 'static + Send;
        let _: impl ?Sized;
        let _: impl ~const Clone;
        let _: impl for<'a> Send;
    }

    /// TyKind::Paren
    fn ty_paren() {
        let _: (T);
    }

    /// TyKind::Typeof
    fn ty_typeof() {
        /*! unused for now */
    }

    /// TyKind::Infer
    fn ty_infer() {
        let _: _;
    }

    /// TyKind::ImplicitSelf
    fn ty_implicit_self() {
        /*! there is no syntax for this */
    }

    /// TyKind::MacCall
    fn ty_mac_call() {
        let _: concat_idents!(T);
        let _: concat_idents![T];
        let _: concat_idents! { T };
    }

    /// TyKind::CVarArgs
    fn ty_c_var_args() {
        /*! FIXME: todo */
    }

    /// TyKind::Pat
    fn ty_pat() {
        let _: core::pattern_type!(u32 is 1..);
    }
}

mod visibilities {
    /// VisibilityKind::Public
    mod visibility_public {
        pub struct Pub;
    }

    /// VisibilityKind::Restricted
    mod visibility_restricted {
        pub(crate) struct PubCrate;
        pub(self) struct PubSelf;
        pub(super) struct PubSuper;
        pub(in crate) struct PubInCrate;
        pub(in self) struct PubInSelf;
        pub(in super) struct PubInSuper;
        pub(in crate::visibilities) struct PubInCrateVisibilities;
        pub(in self::super) struct PubInSelfSuper;
        pub(in super::visibility_restricted) struct PubInSuperMod;
    }
}
