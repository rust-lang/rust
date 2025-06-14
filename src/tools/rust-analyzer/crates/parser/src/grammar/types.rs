use super::*;

pub(super) const TYPE_FIRST: TokenSet = paths::PATH_FIRST.union(TokenSet::new(&[
    T!['('],
    T!['['],
    T![<],
    T![!],
    T![*],
    T![&],
    T![_],
    T![fn],
    T![unsafe],
    T![extern],
    T![for],
    T![impl],
    T![dyn],
    T![Self],
    LIFETIME_IDENT,
]));

pub(super) const TYPE_RECOVERY_SET: TokenSet = TokenSet::new(&[
    T![')'],
    // test_err type_in_array_recover
    // const _: [&];
    T![']'],
    T!['}'],
    T![>],
    T![,],
    // test_err struct_field_recover
    // struct S { f pub g: () }
    // struct S { f: pub g: () }
    T![pub],
]);

pub(crate) fn type_(p: &mut Parser<'_>) {
    type_with_bounds_cond(p, true);
}

pub(super) fn type_no_bounds(p: &mut Parser<'_>) {
    type_with_bounds_cond(p, false);
}

fn type_with_bounds_cond(p: &mut Parser<'_>, allow_bounds: bool) {
    match p.current() {
        T!['('] => paren_or_tuple_type(p),
        T![!] => never_type(p),
        T![*] => ptr_type(p),
        T!['['] => array_or_slice_type(p),
        T![&] => ref_type(p),
        T![_] => infer_type(p),
        T![fn] | T![unsafe] | T![extern] => fn_ptr_type(p),
        T![for] => for_type(p, allow_bounds),
        T![impl] => impl_trait_type(p),
        T![dyn] => dyn_trait_type(p),
        // Some path types are not allowed to have bounds (no plus)
        T![<] => path_type_bounds(p, allow_bounds),
        T![ident] if !p.edition().at_least_2018() && is_dyn_weak(p) => dyn_trait_type_weak(p),
        _ if paths::is_path_start(p) => path_or_macro_type(p, allow_bounds),
        LIFETIME_IDENT if p.nth_at(1, T![+]) => bare_dyn_trait_type(p),
        _ => {
            p.err_recover("expected type", TYPE_RECOVERY_SET);
        }
    }
}

pub(crate) fn is_dyn_weak(p: &Parser<'_>) -> bool {
    const WEAK_DYN_PATH_FIRST: TokenSet = TokenSet::new(&[
        IDENT,
        T![self],
        T![super],
        T![crate],
        T![Self],
        T![lifetime_ident],
        T![?],
        T![for],
        T!['('],
    ]);

    p.at_contextual_kw(T![dyn]) && {
        let la = p.nth(1);
        WEAK_DYN_PATH_FIRST.contains(la) && (la != T![:] || la != T![<])
    }
}

pub(super) fn ascription(p: &mut Parser<'_>) {
    assert!(p.at(T![:]));
    p.bump(T![:]);
    if p.at(T![=]) {
        // recover from `let x: = expr;`, `const X: = expr;` and similar
        // hopefully no type starts with `=`
        p.error("missing type");
        return;
    }
    type_(p);
}

fn paren_or_tuple_type(p: &mut Parser<'_>) {
    assert!(p.at(T!['(']));
    let m = p.start();
    p.bump(T!['(']);
    let mut n_types: u32 = 0;
    let mut trailing_comma: bool = false;
    while !p.at(EOF) && !p.at(T![')']) {
        n_types += 1;
        type_(p);
        if p.eat(T![,]) {
            trailing_comma = true;
        } else {
            trailing_comma = false;
            break;
        }
    }
    p.expect(T![')']);

    let kind = if n_types == 1 && !trailing_comma {
        // test paren_type
        // type T = (i32);
        PAREN_TYPE
    } else {
        // test unit_type
        // type T = ();

        // test singleton_tuple_type
        // type T = (i32,);
        TUPLE_TYPE
    };
    m.complete(p, kind);
}

// test never_type
// type Never = !;
fn never_type(p: &mut Parser<'_>) {
    assert!(p.at(T![!]));
    let m = p.start();
    p.bump(T![!]);
    m.complete(p, NEVER_TYPE);
}

fn ptr_type(p: &mut Parser<'_>) {
    assert!(p.at(T![*]));
    let m = p.start();
    p.bump(T![*]);

    match p.current() {
        // test pointer_type_mut
        // type M = *mut ();
        // type C = *mut ();
        T![mut] | T![const] => p.bump_any(),
        _ => {
            // test_err pointer_type_no_mutability
            // type T = *();
            p.error(
                "expected mut or const in raw pointer type \
                 (use `*mut T` or `*const T` as appropriate)",
            );
        }
    };

    type_no_bounds(p);
    m.complete(p, PTR_TYPE);
}

fn array_or_slice_type(p: &mut Parser<'_>) {
    assert!(p.at(T!['[']));
    let m = p.start();
    p.bump(T!['[']);

    type_(p);
    let kind = match p.current() {
        // test slice_type
        // type T = [()];
        T![']'] => {
            p.bump(T![']']);
            SLICE_TYPE
        }

        // test array_type
        // type T = [(); 92];
        T![;] => {
            p.bump(T![;]);
            let m = p.start();
            expressions::expr(p);
            m.complete(p, CONST_ARG);
            p.expect(T![']']);
            ARRAY_TYPE
        }
        // test_err array_type_missing_semi
        // type T = [() 92];
        _ => {
            p.error("expected `;` or `]`");
            SLICE_TYPE
        }
    };
    m.complete(p, kind);
}

// test reference_type
// type A = &();
// type B = &'static ();
// type C = &mut ();
fn ref_type(p: &mut Parser<'_>) {
    assert!(p.at(T![&]));
    let m = p.start();
    p.bump(T![&]);
    if p.at(LIFETIME_IDENT) {
        lifetime(p);
    }
    p.eat(T![mut]);
    type_no_bounds(p);
    m.complete(p, REF_TYPE);
}

// test placeholder_type
// type Placeholder = _;
fn infer_type(p: &mut Parser<'_>) {
    assert!(p.at(T![_]));
    let m = p.start();
    p.bump(T![_]);
    m.complete(p, INFER_TYPE);
}

// test fn_pointer_type
// type A = fn();
// type B = unsafe fn();
// type C = unsafe extern "C" fn();
// type D = extern "C" fn ( u8 , ... ) -> u8;
fn fn_ptr_type(p: &mut Parser<'_>) {
    let m = p.start();
    p.eat(T![unsafe]);
    if p.at(T![extern]) {
        abi(p);
    }
    // test_err fn_pointer_type_missing_fn
    // type F = unsafe ();
    if !p.eat(T![fn]) {
        m.abandon(p);
        p.error("expected `fn`");
        return;
    }
    if p.at(T!['(']) {
        params::param_list_fn_ptr(p);
    } else {
        p.error("expected parameters");
    }
    // test fn_pointer_type_with_ret
    // type F = fn() -> ();
    opt_ret_type(p);
    m.complete(p, FN_PTR_TYPE);
}

pub(super) fn for_binder(p: &mut Parser<'_>) {
    assert!(p.at(T![for]));
    p.bump(T![for]);
    if p.at(T![<]) {
        generic_params::opt_generic_param_list(p);
    } else {
        p.error("expected `<`");
    }
}

// test for_type
// type A = for<'a> fn() -> ();
// type B = for<'a> unsafe extern "C" fn(&'a ()) -> ();
// type Obj = for<'a> PartialEq<&'a i32>;
pub(super) fn for_type(p: &mut Parser<'_>, allow_bounds: bool) {
    assert!(p.at(T![for]));
    let m = p.start();
    for_binder(p);
    match p.current() {
        T![fn] | T![unsafe] | T![extern] => {}
        // OK: legacy trait object format
        _ if paths::is_use_path_start(p) => {}
        _ => {
            p.error("expected a function pointer or path");
        }
    }
    type_no_bounds(p);
    let completed = m.complete(p, FOR_TYPE);

    // test no_dyn_trait_leading_for
    // type A = for<'a> Test<'a> + Send;
    if allow_bounds {
        opt_type_bounds_as_dyn_trait_type(p, completed);
    }
}

// test impl_trait_type
// type A = impl Iterator<Item=Foo<'a>> + 'a;
fn impl_trait_type(p: &mut Parser<'_>) {
    assert!(p.at(T![impl]));
    let m = p.start();
    p.bump(T![impl]);
    generic_params::bounds_without_colon(p);
    m.complete(p, IMPL_TRAIT_TYPE);
}

// test dyn_trait_type
// type A = dyn Iterator<Item=Foo<'a>> + 'a;
fn dyn_trait_type(p: &mut Parser<'_>) {
    assert!(p.at(T![dyn]));
    let m = p.start();
    p.bump(T![dyn]);
    generic_params::bounds_without_colon(p);
    m.complete(p, DYN_TRAIT_TYPE);
}

// test dyn_trait_type_weak 2015
// type DynPlain = dyn Path;
// type DynRef = &dyn Path;
// type DynLt = dyn 'a + Path;
// type DynQuestion = dyn ?Path;
// type DynFor = dyn for<'a> Path;
// type DynParen = dyn(Path);
// type Path = dyn::Path;
// type Generic = dyn<Path>;
fn dyn_trait_type_weak(p: &mut Parser<'_>) {
    assert!(p.at_contextual_kw(T![dyn]));
    let m = p.start();
    p.bump_remap(T![dyn]);
    generic_params::bounds_without_colon(p);
    m.complete(p, DYN_TRAIT_TYPE);
}

// test bare_dyn_types_with_leading_lifetime
// type A = 'static + Trait;
// type B = S<'static + Trait>;
fn bare_dyn_trait_type(p: &mut Parser<'_>) {
    let m = p.start();
    generic_params::bounds_without_colon(p);
    m.complete(p, DYN_TRAIT_TYPE);
}

// test macro_call_type
// type A = foo!();
// type B = crate::foo!();
fn path_or_macro_type(p: &mut Parser<'_>, allow_bounds: bool) {
    assert!(paths::is_path_start(p));
    let r = p.start();
    let m = p.start();

    paths::type_path(p);

    let kind = if p.at(T![!]) && !p.at(T![!=]) {
        items::macro_call_after_excl(p);
        m.complete(p, MACRO_CALL);
        MACRO_TYPE
    } else {
        m.abandon(p);
        PATH_TYPE
    };

    let path = r.complete(p, kind);

    if allow_bounds {
        opt_type_bounds_as_dyn_trait_type(p, path);
    }
}

// test path_type
// type A = Foo;
// type B = ::Foo;
// type C = self::Foo;
// type D = super::Foo;
pub(super) fn path_type_bounds(p: &mut Parser<'_>, allow_bounds: bool) {
    assert!(paths::is_path_start(p));
    let m = p.start();
    paths::type_path(p);

    // test path_type_with_bounds
    // fn foo() -> Box<T + 'f> {}
    // fn foo() -> Box<dyn T + 'f> {}
    let path = m.complete(p, PATH_TYPE);
    if allow_bounds {
        opt_type_bounds_as_dyn_trait_type(p, path);
    }
}

/// This turns a parsed PATH_TYPE or FOR_TYPE optionally into a DYN_TRAIT_TYPE
/// with a TYPE_BOUND_LIST
pub(super) fn opt_type_bounds_as_dyn_trait_type(
    p: &mut Parser<'_>,
    type_marker: CompletedMarker,
) -> CompletedMarker {
    assert!(matches!(
        type_marker.kind(),
        SyntaxKind::PATH_TYPE | SyntaxKind::FOR_TYPE | SyntaxKind::MACRO_TYPE
    ));
    if !p.at(T![+]) {
        return type_marker;
    }

    // First create a TYPE_BOUND from the completed PATH_TYPE
    let m = type_marker.precede(p).complete(p, TYPE_BOUND);

    // Next setup a marker for the TYPE_BOUND_LIST
    let m = m.precede(p);

    // This gets consumed here so it gets properly set
    // in the TYPE_BOUND_LIST
    p.eat(T![+]);

    // Parse rest of the bounds into the TYPE_BOUND_LIST
    let m = generic_params::bounds_without_colon_m(p, m);

    // Finally precede everything with DYN_TRAIT_TYPE
    m.precede(p).complete(p, DYN_TRAIT_TYPE)
}
