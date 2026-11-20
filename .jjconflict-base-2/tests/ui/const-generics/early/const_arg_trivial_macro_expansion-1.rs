//@ known-bug: #132647
//@ dont-check-compiler-stderr
#![allow(unused_braces)]

// FIXME(bootstrap): This isn't a known bug, we just don't want to write any error annotations.
// this is hard because macro expansion errors have their span be inside the *definition* of the
// macro rather than the line *invoking* it. This means we would wind up with hundreds of error
// annotations on the macro definitions below rather than on any of the actual lines
// that act as a "test".
//
// It's also made more complicated by the fact that compiletest generates "extra" expected
// notes to give an assertable macro backtrace as otherwise there would *nothing* to annotate
// on the actual test lines. All of these extra notes result in needing to write hundreds of
// unnecessary notes on almost every line in this file.
//
// Even though this is marked `known-bug` it should still fail if this test starts ICEing which
// is "enough" in this case.

// Test that we correctly create definitions for anon consts even when
// the trivial-ness of the expression is obscured by macro expansions.
//
// Acts as a regression test for: #131915 130321 128016

// macros expanding to idents

macro_rules! unbraced_ident {
    () => {
        ident
    };
}

macro_rules! braced_ident {
    () => {{ ident }};
}

macro_rules! unbraced_unbraced_ident {
    () => {
        unbraced_ident!()
    };
}

macro_rules! braced_unbraced_ident {
    () => {{ unbraced_ident!() }};
}

macro_rules! unbraced_braced_ident {
    () => {
        braced_ident!()
    };
}

macro_rules! braced_braced_ident {
    () => {{ braced_ident!() }};
}

// macros expanding to complex expr

macro_rules! unbraced_expr {
    () => {
        ident.other
    };
}

macro_rules! braced_expr {
    () => {{ ident.otherent }};
}

macro_rules! unbraced_unbraced_expr {
    () => {
        unbraced_expr!()
    };
}

macro_rules! braced_unbraced_expr {
    () => {{ unbraced_expr!() }};
}

macro_rules! unbraced_braced_expr {
    () => {
        braced_expr!()
    };
}

macro_rules! braced_braced_expr {
    () => {{ braced_expr!() }};
}

macro_rules! closure {
    () => { |()| () };
}

macro_rules! empty {
    () => {};
}

#[rustfmt::skip]
mod array_paren_call {
    // Arrays where the expanded result is a `Res::Err`
    fn array_0() -> [(); unbraced_unbraced_ident!()] { loop {} }
    fn array_1() -> [(); braced_unbraced_ident!()] { loop {} }
    fn array_2() -> [(); unbraced_braced_ident!()] { loop {} }
    fn array_3() -> [(); braced_braced_ident!()] { loop {} }
    fn array_4() -> [(); { unbraced_unbraced_ident!() }] { loop {} }
    fn array_5() -> [(); { braced_unbraced_ident!() }] { loop {} }
    fn array_6() -> [(); { unbraced_braced_ident!() }] { loop {} }
    fn array_7() -> [(); { braced_braced_ident!() }] { loop {} }
    fn array_8() -> [(); unbraced_ident!()] { loop {} }
    fn array_9() -> [(); braced_ident!()] { loop {} }
    fn array_10() -> [(); { unbraced_ident!() }] { loop {} }
    fn array_11() -> [(); { braced_ident!() }] { loop {} }

    // Arrays where the expanded result is a `Res::ConstParam`
    fn array_12<const ident: usize>() -> [(); unbraced_unbraced_ident!()] { loop {} }
    fn array_13<const ident: usize>() -> [(); braced_unbraced_ident!()] { loop {} }
    fn array_14<const ident: usize>() -> [(); unbraced_braced_ident!()] { loop {} }
    fn array_15<const ident: usize>() -> [(); braced_braced_ident!()] { loop {} }
    fn array_16<const ident: usize>() -> [(); { unbraced_unbraced_ident!() }] { loop {} }
    fn array_17<const ident: usize>() -> [(); { braced_unbraced_ident!() }] { loop {} }
    fn array_18<const ident: usize>() -> [(); { unbraced_braced_ident!() }] { loop {} }
    fn array_19<const ident: usize>() -> [(); { braced_braced_ident!() }] { loop {} }
    fn array_20<const ident: usize>() -> [(); unbraced_ident!()] { loop {} }
    fn array_21<const ident: usize>() -> [(); braced_ident!()] { loop {} }
    fn array_22<const ident: usize>() -> [(); { unbraced_ident!() }] { loop {} }
    fn array_23<const ident: usize>() -> [(); { braced_ident!() }] { loop {} }

    // Arrays where the expanded result is a complex expr
    fn array_24() -> [(); unbraced_unbraced_expr!()] { loop {} }
    fn array_25() -> [(); braced_unbraced_expr!()] { loop {} }
    fn array_26() -> [(); unbraced_braced_expr!()] { loop {} }
    fn array_27() -> [(); braced_braced_expr!()] { loop {} }
    fn array_28() -> [(); { unbraced_unbraced_expr!() }] { loop {} }
    fn array_29() -> [(); { braced_unbraced_expr!() }] { loop {} }
    fn array_30() -> [(); { unbraced_braced_expr!() }] { loop {} }
    fn array_31() -> [(); { braced_braced_expr!() }] { loop {} }
    fn array_32() -> [(); unbraced_expr!()] { loop {} }
    fn array_33() -> [(); braced_expr!()] { loop {} }
    fn array_34() -> [(); { unbraced_expr!() }] { loop {} }
    fn array_35() -> [(); { braced_expr!() }] { loop {} }

    // Arrays whose expanded form contains a nested definition
    fn array_36() -> [(); closure!()] { loop {} }
    fn array_37() -> [(); { closure!() }] { loop {} }

    // Arrays whose macro expansion is empty
    fn array_38() -> [(); empty!()] { loop {} }
    fn array_39() -> [(); { empty!() }] { loop {} }
}

#[rustfmt::skip]
mod array_brace_call {
    // Arrays where the expanded result is a `Res::Err`
    fn array_0() -> [(); unbraced_unbraced_ident!{}] { loop {} }
    fn array_1() -> [(); braced_unbraced_ident!{}] { loop {} }
    fn array_2() -> [(); unbraced_braced_ident!{}] { loop {} }
    fn array_3() -> [(); braced_braced_ident!{}] { loop {} }
    fn array_4() -> [(); { unbraced_unbraced_ident!{} }] { loop {} }
    fn array_5() -> [(); { braced_unbraced_ident!{} }] { loop {} }
    fn array_6() -> [(); { unbraced_braced_ident!{} }] { loop {} }
    fn array_7() -> [(); { braced_braced_ident!{} }] { loop {} }
    fn array_8() -> [(); unbraced_ident!{}] { loop {} }
    fn array_9() -> [(); braced_ident!{}] { loop {} }
    fn array_10() -> [(); { unbraced_ident!{} }] { loop {} }
    fn array_11() -> [(); { braced_ident!{} }] { loop {} }

    // Arrays where the expanded result is a `Res::ConstParam`
    fn array_12<const ident: usize>() -> [(); unbraced_unbraced_ident!{}] { loop {} }
    fn array_13<const ident: usize>() -> [(); braced_unbraced_ident!{}] { loop {} }
    fn array_14<const ident: usize>() -> [(); unbraced_braced_ident!{}] { loop {} }
    fn array_15<const ident: usize>() -> [(); braced_braced_ident!{}] { loop {} }
    fn array_16<const ident: usize>() -> [(); { unbraced_unbraced_ident!{} }] { loop {} }
    fn array_17<const ident: usize>() -> [(); { braced_unbraced_ident!{} }] { loop {} }
    fn array_18<const ident: usize>() -> [(); { unbraced_braced_ident!{} }] { loop {} }
    fn array_19<const ident: usize>() -> [(); { braced_braced_ident!{} }] { loop {} }
    fn array_20<const ident: usize>() -> [(); unbraced_ident!{}] { loop {} }
    fn array_21<const ident: usize>() -> [(); braced_ident!{}] { loop {} }
    fn array_22<const ident: usize>() -> [(); { unbraced_ident!{} }] { loop {} }
    fn array_23<const ident: usize>() -> [(); { braced_ident!{} }] { loop {} }

    // Arrays where the expanded result is a complex expr
    fn array_24() -> [(); unbraced_unbraced_expr!{}] { loop {} }
    fn array_25() -> [(); braced_unbraced_expr!{}] { loop {} }
    fn array_26() -> [(); unbraced_braced_expr!{}] { loop {} }
    fn array_27() -> [(); braced_braced_expr!{}] { loop {} }
    fn array_28() -> [(); { unbraced_unbraced_expr!{} }] { loop {} }
    fn array_29() -> [(); { braced_unbraced_expr!{} }] { loop {} }
    fn array_30() -> [(); { unbraced_braced_expr!{} }] { loop {} }
    fn array_31() -> [(); { braced_braced_expr!{} }] { loop {} }
    fn array_32() -> [(); unbraced_expr!{}] { loop {} }
    fn array_33() -> [(); braced_expr!{}] { loop {} }
    fn array_34() -> [(); { unbraced_expr!{} }] { loop {} }
    fn array_35() -> [(); { braced_expr!{} }] { loop {} }

    // Arrays whose expanded form contains a nested definition
    fn array_36() -> [(); closure!{}] { loop {} }
    fn array_37() -> [(); { closure!{} }] { loop {} }

    // Arrays whose macro expansion is empty
    fn array_38() -> [(); empty!{}] { loop {} }
    fn array_39() -> [(); { empty!{} }] { loop {} }
}

#[rustfmt::skip]
mod array_square_call {
    // Arrays where the expanded result is a `Res::Err`
    fn array_0() -> [(); unbraced_unbraced_ident![]] { loop {} }
    fn array_1() -> [(); braced_unbraced_ident![]] { loop {} }
    fn array_2() -> [(); unbraced_braced_ident![]] { loop {} }
    fn array_3() -> [(); braced_braced_ident![]] { loop {} }
    fn array_4() -> [(); { unbraced_unbraced_ident![] }] { loop {} }
    fn array_5() -> [(); { braced_unbraced_ident![] }] { loop {} }
    fn array_6() -> [(); { unbraced_braced_ident![] }] { loop {} }
    fn array_7() -> [(); { braced_braced_ident![] }] { loop {} }
    fn array_8() -> [(); unbraced_ident![]] { loop {} }
    fn array_9() -> [(); braced_ident![]] { loop {} }
    fn array_10() -> [(); { unbraced_ident![] }] { loop {} }
    fn array_11() -> [(); { braced_ident![] }] { loop {} }

    // Arrays where the expanded result is a `Res::ConstParam`
    fn array_12<const ident: usize>() -> [(); unbraced_unbraced_ident![]] { loop {} }
    fn array_13<const ident: usize>() -> [(); braced_unbraced_ident![]] { loop {} }
    fn array_14<const ident: usize>() -> [(); unbraced_braced_ident![]] { loop {} }
    fn array_15<const ident: usize>() -> [(); braced_braced_ident![]] { loop {} }
    fn array_16<const ident: usize>() -> [(); { unbraced_unbraced_ident![] }] { loop {} }
    fn array_17<const ident: usize>() -> [(); { braced_unbraced_ident![] }] { loop {} }
    fn array_18<const ident: usize>() -> [(); { unbraced_braced_ident![] }] { loop {} }
    fn array_19<const ident: usize>() -> [(); { braced_braced_ident![] }] { loop {} }
    fn array_20<const ident: usize>() -> [(); unbraced_ident![]] { loop {} }
    fn array_21<const ident: usize>() -> [(); braced_ident![]] { loop {} }
    fn array_22<const ident: usize>() -> [(); { unbraced_ident![] }] { loop {} }
    fn array_23<const ident: usize>() -> [(); { braced_ident![] }] { loop {} }

    // Arrays where the expanded result is a complex expr
    fn array_24() -> [(); unbraced_unbraced_expr![]] { loop {} }
    fn array_25() -> [(); braced_unbraced_expr![]] { loop {} }
    fn array_26() -> [(); unbraced_braced_expr![]] { loop {} }
    fn array_27() -> [(); braced_braced_expr![]] { loop {} }
    fn array_28() -> [(); { unbraced_unbraced_expr![] }] { loop {} }
    fn array_29() -> [(); { braced_unbraced_expr![] }] { loop {} }
    fn array_30() -> [(); { unbraced_braced_expr![] }] { loop {} }
    fn array_31() -> [(); { braced_braced_expr![] }] { loop {} }
    fn array_32() -> [(); unbraced_expr![]] { loop {} }
    fn array_33() -> [(); braced_expr![]] { loop {} }
    fn array_34() -> [(); { unbraced_expr![] }] { loop {} }
    fn array_35() -> [(); { braced_expr![] }] { loop {} }

    // Arrays whose expanded form contains a nested definition
    fn array_36() -> [(); closure![]] { loop {} }
    fn array_37() -> [(); { closure![] }] { loop {} }

    // Arrays whose macro expansion is empty
    fn array_38() -> [(); empty![]] { loop {} }
    fn array_39() -> [(); { empty![] }] { loop {} }
}

struct Foo<const N: usize>;

#[rustfmt::skip]
mod adt_paren_call {
    use super::Foo;

    // An ADT where the expanded result is a `Res::Err`
    fn adt_0() -> Foo<unbraced_unbraced_ident!()> { loop {} }
    fn adt_1() -> Foo<braced_unbraced_ident!()> { loop {} }
    fn adt_2() -> Foo<unbraced_braced_ident!()> { loop {} }
    fn adt_3() -> Foo<braced_braced_ident!()> { loop {} }
    fn adt_4() -> Foo<{ unbraced_unbraced_ident!() }> { loop {} }
    fn adt_5() -> Foo<{ braced_unbraced_ident!() }> { loop {} }
    fn adt_6() -> Foo<{ unbraced_braced_ident!() }> { loop {} }
    fn adt_7() -> Foo<{ braced_braced_ident!() }> { loop {} }
    fn adt_8() -> Foo<unbraced_ident!()> { loop {} }
    fn adt_9() -> Foo<braced_ident!()> { loop {} }
    fn adt_10() -> Foo<{ unbraced_ident!() }> { loop {} }
    fn adt_11() -> Foo<{ braced_ident!() }> { loop {} }

    // An ADT where the expanded result is a `Res::ConstParam`
    fn adt_12<const ident: usize>() -> Foo<unbraced_unbraced_ident!()> { loop {} }
    fn adt_13<const ident: usize>() -> Foo<braced_unbraced_ident!()> { loop {} }
    fn adt_14<const ident: usize>() -> Foo<unbraced_braced_ident!()> { loop {} }
    fn adt_15<const ident: usize>() -> Foo<braced_braced_ident!()> { loop {} }
    fn adt_16<const ident: usize>() -> Foo<{ unbraced_unbraced_ident!() }> { loop {} }
    fn adt_17<const ident: usize>() -> Foo<{ braced_unbraced_ident!() }> { loop {} }
    fn adt_18<const ident: usize>() -> Foo<{ unbraced_braced_ident!() }> { loop {} }
    fn adt_19<const ident: usize>() -> Foo<{ braced_braced_ident!() }> { loop {} }
    fn adt_20<const ident: usize>() -> Foo<unbraced_ident!()> { loop {} }
    fn adt_21<const ident: usize>() -> Foo<braced_ident!()> { loop {} }
    fn adt_22<const ident: usize>() -> Foo<{ unbraced_ident!() }> { loop {} }
    fn adt_23<const ident: usize>() -> Foo<{ braced_ident!() }> { loop {} }

    // An ADT where the expanded result is a complex expr
    fn adt_24() -> Foo<unbraced_unbraced_expr!()> { loop {} }
    fn adt_25() -> Foo<braced_unbraced_expr!()> { loop {} }
    fn adt_26() -> Foo<unbraced_braced_expr!()> { loop {} }
    fn adt_27() -> Foo<braced_braced_expr!()> { loop {} }
    fn adt_28() -> Foo<{ unbraced_unbraced_expr!() }> { loop {} }
    fn adt_29() -> Foo<{ braced_unbraced_expr!() }> { loop {} }
    fn adt_30() -> Foo<{ unbraced_braced_expr!() }> { loop {} }
    fn adt_31() -> Foo<{ braced_braced_expr!() }> { loop {} }
    fn adt_32() -> Foo<unbraced_expr!()> { loop {} }
    fn adt_33() -> Foo<braced_expr!()> { loop {} }
    fn adt_34() -> Foo<{ unbraced_expr!() }> { loop {} }
    fn adt_35() -> Foo<{ braced_expr!() }> { loop {} }

    // An ADT whose expanded form contains a nested definition
    fn adt_36() -> Foo<closure!()> { loop {} }
    fn adt_37() -> Foo<{ closure!() }> { loop {} }

    // An ADT whose macro expansion is empty
    fn adt_38() -> Foo<empty!()> { loop {} }
    fn adt_39() -> Foo<{ empty!() }> { loop {} }
}

#[rustfmt::skip]
mod adt_brace_call {
    use super::Foo;

    // An ADT where the expanded result is a `Res::Err`
    fn adt_0() -> Foo<unbraced_unbraced_ident!{}> { loop {} }
    fn adt_1() -> Foo<braced_unbraced_ident!{}> { loop {} }
    fn adt_2() -> Foo<unbraced_braced_ident!{}> { loop {} }
    fn adt_3() -> Foo<braced_braced_ident!{}> { loop {} }
    fn adt_4() -> Foo<{ unbraced_unbraced_ident!{} }> { loop {} }
    fn adt_5() -> Foo<{ braced_unbraced_ident!{} }> { loop {} }
    fn adt_6() -> Foo<{ unbraced_braced_ident!{} }> { loop {} }
    fn adt_7() -> Foo<{ braced_braced_ident!{} }> { loop {} }
    fn adt_8() -> Foo<unbraced_ident!{}> { loop {} }
    fn adt_9() -> Foo<braced_ident!{}> { loop {} }
    fn adt_10() -> Foo<{ unbraced_ident!{} }> { loop {} }
    fn adt_11() -> Foo<{ braced_ident!{} }> { loop {} }

    // An ADT where the expanded result is a `Res::ConstParam`
    fn adt_12<const ident: usize>() -> Foo<unbraced_unbraced_ident!{}> { loop {} }
    fn adt_13<const ident: usize>() -> Foo<braced_unbraced_ident!{}> { loop {} }
    fn adt_14<const ident: usize>() -> Foo<unbraced_braced_ident!{}> { loop {} }
    fn adt_15<const ident: usize>() -> Foo<braced_braced_ident!{}> { loop {} }
    fn adt_16<const ident: usize>() -> Foo<{ unbraced_unbraced_ident!{} }> { loop {} }
    fn adt_17<const ident: usize>() -> Foo<{ braced_unbraced_ident!{} }> { loop {} }
    fn adt_18<const ident: usize>() -> Foo<{ unbraced_braced_ident!{} }> { loop {} }
    fn adt_19<const ident: usize>() -> Foo<{ braced_braced_ident!{} }> { loop {} }
    fn adt_20<const ident: usize>() -> Foo<unbraced_ident!{}> { loop {} }
    fn adt_21<const ident: usize>() -> Foo<braced_ident!{}> { loop {} }
    fn adt_22<const ident: usize>() -> Foo<{ unbraced_ident!{} }> { loop {} }
    fn adt_23<const ident: usize>() -> Foo<{ braced_ident!{} }> { loop {} }

    // An ADT where the expanded result is a complex expr
    fn adt_24() -> Foo<unbraced_unbraced_expr!{}> { loop {} }
    fn adt_25() -> Foo<braced_unbraced_expr!{}> { loop {} }
    fn adt_26() -> Foo<unbraced_braced_expr!{}> { loop {} }
    fn adt_27() -> Foo<braced_braced_expr!{}> { loop {} }
    fn adt_28() -> Foo<{ unbraced_unbraced_expr!{} }> { loop {} }
    fn adt_29() -> Foo<{ braced_unbraced_expr!{} }> { loop {} }
    fn adt_30() -> Foo<{ unbraced_braced_expr!{} }> { loop {} }
    fn adt_31() -> Foo<{ braced_braced_expr!{} }> { loop {} }
    fn adt_32() -> Foo<unbraced_expr!{}> { loop {} }
    fn adt_33() -> Foo<braced_expr!{}> { loop {} }
    fn adt_34() -> Foo<{ unbraced_expr!{} }> { loop {} }
    fn adt_35() -> Foo<{ braced_expr!{} }> { loop {} }

    // An ADT whose expanded form contains a nested definition
    fn adt_36() -> Foo<closure!{}> { loop {} }
    fn adt_37() -> Foo<{ closure!{} }> { loop {} }

    // An ADT whose macro expansion is empty
    fn adt_38() -> Foo<empty!{}> { loop {} }
    fn adt_39() -> Foo<{ empty!{} }> { loop {} }
}

#[rustfmt::skip]
mod adt_square_call {
    use super::Foo;

    // An ADT where the expanded result is a `Res::Err`
    fn adt_0() -> Foo<unbraced_unbraced_ident![]> { loop {} }
    fn adt_1() -> Foo<braced_unbraced_ident![]> { loop {} }
    fn adt_2() -> Foo<unbraced_braced_ident![]> { loop {} }
    fn adt_3() -> Foo<braced_braced_ident![]> { loop {} }
    fn adt_4() -> Foo<{ unbraced_unbraced_ident![] }> { loop {} }
    fn adt_5() -> Foo<{ braced_unbraced_ident![] }> { loop {} }
    fn adt_6() -> Foo<{ unbraced_braced_ident![] }> { loop {} }
    fn adt_7() -> Foo<{ braced_braced_ident![] }> { loop {} }
    fn adt_8() -> Foo<unbraced_ident![]> { loop {} }
    fn adt_9() -> Foo<braced_ident![]> { loop {} }
    fn adt_10() -> Foo<{ unbraced_ident![] }> { loop {} }
    fn adt_11() -> Foo<{ braced_ident![] }> { loop {} }

    // An ADT where the expanded result is a `Res::ConstParam`
    fn adt_12<const ident: usize>() -> Foo<unbraced_unbraced_ident![]> { loop {} }
    fn adt_13<const ident: usize>() -> Foo<braced_unbraced_ident![]> { loop {} }
    fn adt_14<const ident: usize>() -> Foo<unbraced_braced_ident![]> { loop {} }
    fn adt_15<const ident: usize>() -> Foo<braced_braced_ident![]> { loop {} }
    fn adt_16<const ident: usize>() -> Foo<{ unbraced_unbraced_ident![] }> { loop {} }
    fn adt_17<const ident: usize>() -> Foo<{ braced_unbraced_ident![] }> { loop {} }
    fn adt_18<const ident: usize>() -> Foo<{ unbraced_braced_ident![] }> { loop {} }
    fn adt_19<const ident: usize>() -> Foo<{ braced_braced_ident![] }> { loop {} }
    fn adt_20<const ident: usize>() -> Foo<unbraced_ident![]> { loop {} }
    fn adt_21<const ident: usize>() -> Foo<braced_ident![]> { loop {} }
    fn adt_22<const ident: usize>() -> Foo<{ unbraced_ident![] }> { loop {} }
    fn adt_23<const ident: usize>() -> Foo<{ braced_ident![] }> { loop {} }

    // An ADT where the expanded result is a complex expr
    fn adt_24() -> Foo<unbraced_unbraced_expr![]> { loop {} }
    fn adt_25() -> Foo<braced_unbraced_expr![]> { loop {} }
    fn adt_26() -> Foo<unbraced_braced_expr![]> { loop {} }
    fn adt_27() -> Foo<braced_braced_expr![]> { loop {} }
    fn adt_28() -> Foo<{ unbraced_unbraced_expr![] }> { loop {} }
    fn adt_29() -> Foo<{ braced_unbraced_expr![] }> { loop {} }
    fn adt_30() -> Foo<{ unbraced_braced_expr![] }> { loop {} }
    fn adt_31() -> Foo<{ braced_braced_expr![] }> { loop {} }
    fn adt_32() -> Foo<unbraced_expr![]> { loop {} }
    fn adt_33() -> Foo<braced_expr![]> { loop {} }
    fn adt_34() -> Foo<{ unbraced_expr![] }> { loop {} }
    fn adt_35() -> Foo<{ braced_expr![] }> { loop {} }

    // An ADT whose expanded form contains a nested definition
    fn adt_36() -> Foo<closure![]> { loop {} }
    fn adt_37() -> Foo<{ closure![] }> { loop {} }

    // An ADT whose macro expansion is empty
    fn adt_38() -> Foo<empty![]> { loop {} }
    fn adt_39() -> Foo<{ empty![] }> { loop {} }
}

#[rustfmt::skip]
mod repeat_paren_call {
    // A repeat expr where the expanded result is a `Res::Err`
    fn repeat_0() { [(); unbraced_unbraced_ident!()]; }
    fn repeat_1() { [(); braced_unbraced_ident!()]; }
    fn repeat_2() { [(); unbraced_braced_ident!()]; }
    fn repeat_3() { [(); braced_braced_ident!()]; }
    fn repeat_4() { [(); { unbraced_unbraced_ident!() }]; }
    fn repeat_5() { [(); { braced_unbraced_ident!() }]; }
    fn repeat_6() { [(); { unbraced_braced_ident!() }]; }
    fn repeat_7() { [(); { braced_braced_ident!() }]; }
    fn repeat_8() { [(); unbraced_ident!()]; }
    fn repeat_9() { [(); braced_ident!()]; }
    fn repeat_10() { [(); { unbraced_ident!() }]; }
    fn repeat_11() { [(); { braced_ident!() }]; }

    // A repeat expr where the expanded result is a `Res::ConstParam`
    fn repeat_12<const ident: usize>() { [(); unbraced_unbraced_ident!()]; }
    fn repeat_13<const ident: usize>() { [(); braced_unbraced_ident!()]; }
    fn repeat_14<const ident: usize>() { [(); unbraced_braced_ident!()]; }
    fn repeat_15<const ident: usize>() { [(); braced_braced_ident!()]; }
    fn repeat_16<const ident: usize>() { [(); { unbraced_unbraced_ident!() }]; }
    fn repeat_17<const ident: usize>() { [(); { braced_unbraced_ident!() }]; }
    fn repeat_18<const ident: usize>() { [(); { unbraced_braced_ident!() }]; }
    fn repeat_19<const ident: usize>() { [(); { braced_braced_ident!() }]; }
    fn repeat_20<const ident: usize>() { [(); unbraced_ident!()]; }
    fn repeat_21<const ident: usize>() { [(); braced_ident!()]; }
    fn repeat_22<const ident: usize>() { [(); { unbraced_ident!() }]; }
    fn repeat_23<const ident: usize>() { [(); { braced_ident!() }]; }

    // A repeat expr where the expanded result is a complex expr
    fn repeat_24() { [(); unbraced_unbraced_expr!()]; }
    fn repeat_25() { [(); braced_unbraced_expr!()]; }
    fn repeat_26() { [(); unbraced_braced_expr!()]; }
    fn repeat_27() { [(); braced_braced_expr!()]; }
    fn repeat_28() { [(); { unbraced_unbraced_expr!() }]; }
    fn repeat_29() { [(); { braced_unbraced_expr!() }]; }
    fn repeat_30() { [(); { unbraced_braced_expr!() }]; }
    fn repeat_31() { [(); { braced_braced_expr!() }]; }
    fn repeat_32() { [(); unbraced_expr!()]; }
    fn repeat_33() { [(); braced_expr!()]; }
    fn repeat_34() { [(); { unbraced_expr!() }]; }
    fn repeat_35() { [(); { braced_expr!() }]; }

    // A repeat expr whose expanded form contains a nested definition
    fn repeat_36() { [(); closure!()] }
    fn repeat_37() { [(); { closure!() }] }

    // A repeat expr whose macro expansion is empty
    fn repeat_38() { [(); empty!()] }
    fn repeat_39() { [(); { empty!() }] }
}

#[rustfmt::skip]
mod repeat_brace_call {
    // A repeat expr where the expanded result is a `Res::Err`
    fn repeat_0() { [(); unbraced_unbraced_ident!{}]; }
    fn repeat_1() { [(); braced_unbraced_ident!{}]; }
    fn repeat_2() { [(); unbraced_braced_ident!{}]; }
    fn repeat_3() { [(); braced_braced_ident!{}]; }
    fn repeat_4() { [(); { unbraced_unbraced_ident!{} }]; }
    fn repeat_5() { [(); { braced_unbraced_ident!{} }]; }
    fn repeat_6() { [(); { unbraced_braced_ident!{} }]; }
    fn repeat_7() { [(); { braced_braced_ident!{} }]; }
    fn repeat_8() { [(); unbraced_ident!{}]; }
    fn repeat_9() { [(); braced_ident!{}]; }
    fn repeat_10() { [(); { unbraced_ident!{} }]; }
    fn repeat_11() { [(); { braced_ident!{} }]; }

    // A repeat expr where the expanded result is a `Res::ConstParam`
    fn repeat_12<const ident: usize>() { [(); unbraced_unbraced_ident!{}]; }
    fn repeat_13<const ident: usize>() { [(); braced_unbraced_ident!{}]; }
    fn repeat_14<const ident: usize>() { [(); unbraced_braced_ident!{}]; }
    fn repeat_15<const ident: usize>() { [(); braced_braced_ident!{}]; }
    fn repeat_16<const ident: usize>() { [(); { unbraced_unbraced_ident!{} }]; }
    fn repeat_17<const ident: usize>() { [(); { braced_unbraced_ident!{} }]; }
    fn repeat_18<const ident: usize>() { [(); { unbraced_braced_ident!{} }]; }
    fn repeat_19<const ident: usize>() { [(); { braced_braced_ident!{} }]; }
    fn repeat_20<const ident: usize>() { [(); unbraced_ident!{}]; }
    fn repeat_21<const ident: usize>() { [(); braced_ident!{}]; }
    fn repeat_22<const ident: usize>() { [(); { unbraced_ident!{} }]; }
    fn repeat_23<const ident: usize>() { [(); { braced_ident!{} }]; }

    // A repeat expr where the expanded result is a complex expr
    fn repeat_24() { [(); unbraced_unbraced_expr!{}]; }
    fn repeat_25() { [(); braced_unbraced_expr!{}]; }
    fn repeat_26() { [(); unbraced_braced_expr!{}]; }
    fn repeat_27() { [(); braced_braced_expr!{}]; }
    fn repeat_28() { [(); { unbraced_unbraced_expr!{} }]; }
    fn repeat_29() { [(); { braced_unbraced_expr!{} }]; }
    fn repeat_30() { [(); { unbraced_braced_expr!{} }]; }
    fn repeat_31() { [(); { braced_braced_expr!{} }]; }
    fn repeat_32() { [(); unbraced_expr!{}]; }
    fn repeat_33() { [(); braced_expr!{}]; }
    fn repeat_34() { [(); { unbraced_expr!{} }]; }
    fn repeat_35() { [(); { braced_expr!{} }]; }

    // A repeat expr whose expanded form contains a nested definition
    fn repeat_36() { [(); closure!{}] }
    fn repeat_37() { [(); { closure!{} }] }

    // A repeat expr whose macro expansion is empty
    fn repeat_38() { [(); empty!{}] }
    fn repeat_39() { [(); { empty!{} }] }
}

#[rustfmt::skip]
mod repeat_square_call {
    // A repeat expr where the expanded result is a `Res::Err`
    fn repeat_0() { [(); unbraced_unbraced_ident![]]; }
    fn repeat_1() { [(); braced_unbraced_ident![]]; }
    fn repeat_2() { [(); unbraced_braced_ident![]]; }
    fn repeat_3() { [(); braced_braced_ident![]]; }
    fn repeat_4() { [(); { unbraced_unbraced_ident![] }]; }
    fn repeat_5() { [(); { braced_unbraced_ident![] }]; }
    fn repeat_6() { [(); { unbraced_braced_ident![] }]; }
    fn repeat_7() { [(); { braced_braced_ident![] }]; }
    fn repeat_8() { [(); unbraced_ident![]]; }
    fn repeat_9() { [(); braced_ident![]]; }
    fn repeat_10() { [(); { unbraced_ident![] }]; }
    fn repeat_11() { [(); { braced_ident![] }]; }

    // A repeat expr where the expanded result is a `Res::ConstParam`
    fn repeat_12<const ident: usize>() { [(); unbraced_unbraced_ident![]]; }
    fn repeat_13<const ident: usize>() { [(); braced_unbraced_ident![]]; }
    fn repeat_14<const ident: usize>() { [(); unbraced_braced_ident![]]; }
    fn repeat_15<const ident: usize>() { [(); braced_braced_ident![]]; }
    fn repeat_16<const ident: usize>() { [(); { unbraced_unbraced_ident![] }]; }
    fn repeat_17<const ident: usize>() { [(); { braced_unbraced_ident![] }]; }
    fn repeat_18<const ident: usize>() { [(); { unbraced_braced_ident![] }]; }
    fn repeat_19<const ident: usize>() { [(); { braced_braced_ident![] }]; }
    fn repeat_20<const ident: usize>() { [(); unbraced_ident![]]; }
    fn repeat_21<const ident: usize>() { [(); braced_ident![]]; }
    fn repeat_22<const ident: usize>() { [(); { unbraced_ident![] }]; }
    fn repeat_23<const ident: usize>() { [(); { braced_ident![] }]; }

    // A repeat expr where the expanded result is a complex expr
    fn repeat_24() { [(); unbraced_unbraced_expr![]]; }
    fn repeat_25() { [(); braced_unbraced_expr![]]; }
    fn repeat_26() { [(); unbraced_braced_expr![]]; }
    fn repeat_27() { [(); braced_braced_expr![]]; }
    fn repeat_28() { [(); { unbraced_unbraced_expr![] }]; }
    fn repeat_29() { [(); { braced_unbraced_expr![] }]; }
    fn repeat_30() { [(); { unbraced_braced_expr![] }]; }
    fn repeat_31() { [(); { braced_braced_expr![] }]; }
    fn repeat_32() { [(); unbraced_expr![]]; }
    fn repeat_33() { [(); braced_expr![]]; }
    fn repeat_34() { [(); { unbraced_expr![] }]; }
    fn repeat_35() { [(); { braced_expr![] }]; }

    // A repeat expr whose expanded form contains a nested definition
    fn repeat_36() { [(); closure![]] }
    fn repeat_37() { [(); { closure![] }] }

    // A repeat expr whose macro expansion is empty
    fn repeat_38() { [(); empty![]] }
    fn repeat_39() { [(); { empty![] }] }
}

fn main() {}
