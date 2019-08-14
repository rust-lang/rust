use super::*;

pub(super) const TYPE_FIRST: TokenSet = paths::PATH_FIRST.union(token_set![
    L_PAREN, EXCL, STAR, L_BRACK, AMP, UNDERSCORE, FN_KW, UNSAFE_KW, EXTERN_KW, FOR_KW, IMPL_KW,
    DYN_KW, L_ANGLE,
]);

const TYPE_RECOVERY_SET: TokenSet = token_set![R_PAREN, COMMA];

pub(super) fn type_(p: &mut Parser) {
    type_with_bounds_cond(p, true);
}

pub(super) fn type_no_bounds(p: &mut Parser) {
    type_with_bounds_cond(p, false);
}

fn type_with_bounds_cond(p: &mut Parser, allow_bounds: bool) {
    match p.current() {
        T!['('] => paren_or_tuple_type(p),
        T![!] => never_type(p),
        T![*] => pointer_type(p),
        T!['['] => array_or_slice_type(p),
        T![&] => reference_type(p),
        T![_] => placeholder_type(p),
        T![fn] | T![unsafe] | T![extern] => fn_pointer_type(p),
        T![for] => for_type(p),
        T![impl] => impl_trait_type(p),
        T![dyn ] => dyn_trait_type(p),
        // Some path types are not allowed to have bounds (no plus)
        T![<] => path_type_(p, allow_bounds),
        _ if paths::is_use_path_start(p) => path_or_macro_type_(p, allow_bounds),
        _ => {
            p.err_recover("expected type", TYPE_RECOVERY_SET);
        }
    }
}

pub(super) fn ascription(p: &mut Parser) {
    p.expect(T![:]);
    type_(p)
}

fn paren_or_tuple_type(p: &mut Parser) {
    assert!(p.at(T!['(']));
    let m = p.start();
    p.bump();
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
fn never_type(p: &mut Parser) {
    assert!(p.at(T![!]));
    let m = p.start();
    p.bump();
    m.complete(p, NEVER_TYPE);
}

fn pointer_type(p: &mut Parser) {
    assert!(p.at(T![*]));
    let m = p.start();
    p.bump();

    match p.current() {
        // test pointer_type_mut
        // type M = *mut ();
        // type C = *mut ();
        T![mut] | T![const] => p.bump(),
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
    m.complete(p, POINTER_TYPE);
}

fn array_or_slice_type(p: &mut Parser) {
    assert!(p.at(T!['[']));
    let m = p.start();
    p.bump();

    type_(p);
    let kind = match p.current() {
        // test slice_type
        // type T = [()];
        T![']'] => {
            p.bump();
            SLICE_TYPE
        }

        // test array_type
        // type T = [(); 92];
        T![;] => {
            p.bump();
            expressions::expr(p);
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

// test reference_type;
// type A = &();
// type B = &'static ();
// type C = &mut ();
fn reference_type(p: &mut Parser) {
    assert!(p.at(T![&]));
    let m = p.start();
    p.bump();
    p.eat(LIFETIME);
    p.eat(T![mut]);
    type_no_bounds(p);
    m.complete(p, REFERENCE_TYPE);
}

// test placeholder_type
// type Placeholder = _;
fn placeholder_type(p: &mut Parser) {
    assert!(p.at(T![_]));
    let m = p.start();
    p.bump();
    m.complete(p, PLACEHOLDER_TYPE);
}

// test fn_pointer_type
// type A = fn();
// type B = unsafe fn();
// type C = unsafe extern "C" fn();
// type D = extern "C" fn ( u8 , ... ) -> u8;
fn fn_pointer_type(p: &mut Parser) {
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
        params::param_list_opt_patterns(p);
    } else {
        p.error("expected parameters")
    }
    // test fn_pointer_type_with_ret
    // type F = fn() -> ();
    opt_fn_ret_type(p);
    m.complete(p, FN_POINTER_TYPE);
}

pub(super) fn for_binder(p: &mut Parser) {
    assert!(p.at(T![for]));
    p.bump();
    if p.at(T![<]) {
        type_params::opt_type_param_list(p);
    } else {
        p.error("expected `<`");
    }
}

// test for_type
// type A = for<'a> fn() -> ();
// fn foo<T>(_t: &T) where for<'a> &'a T: Iterator {}
// fn bar<T>(_t: &T) where for<'a> &'a mut T: Iterator {}
// fn baz<T>(_t: &T) where for<'a> <&'a T as Baz>::Foo: Iterator {}
pub(super) fn for_type(p: &mut Parser) {
    assert!(p.at(T![for]));
    let m = p.start();
    for_binder(p);
    match p.current() {
        T![fn] | T![unsafe] | T![extern] => fn_pointer_type(p),
        T![&] => reference_type(p),
        _ if paths::is_path_start(p) => path_type_(p, false),
        _ => p.error("expected a path"),
    }
    m.complete(p, FOR_TYPE);
}

// test impl_trait_type
// type A = impl Iterator<Item=Foo<'a>> + 'a;
fn impl_trait_type(p: &mut Parser) {
    assert!(p.at(T![impl]));
    let m = p.start();
    p.bump();
    type_params::bounds_without_colon(p);
    m.complete(p, IMPL_TRAIT_TYPE);
}

// test dyn_trait_type
// type A = dyn Iterator<Item=Foo<'a>> + 'a;
fn dyn_trait_type(p: &mut Parser) {
    assert!(p.at(T![dyn ]));
    let m = p.start();
    p.bump();
    type_params::bounds_without_colon(p);
    m.complete(p, DYN_TRAIT_TYPE);
}

// test path_type
// type A = Foo;
// type B = ::Foo;
// type C = self::Foo;
// type D = super::Foo;
pub(super) fn path_type(p: &mut Parser) {
    path_type_(p, true)
}

// test macro_call_type
// type A = foo!();
// type B = crate::foo!();
fn path_or_macro_type_(p: &mut Parser, allow_bounds: bool) {
    assert!(paths::is_path_start(p));
    let m = p.start();
    paths::type_path(p);

    let kind = if p.at(T![!]) {
        items::macro_call_after_excl(p);
        MACRO_CALL
    } else {
        PATH_TYPE
    };

    let path = m.complete(p, kind);

    if allow_bounds {
        opt_path_type_bounds_as_dyn_trait_type(p, path);
    }
}

pub(super) fn path_type_(p: &mut Parser, allow_bounds: bool) {
    assert!(paths::is_path_start(p));
    let m = p.start();
    paths::type_path(p);

    // test path_type_with_bounds
    // fn foo() -> Box<T + 'f> {}
    // fn foo() -> Box<dyn T + 'f> {}
    let path = m.complete(p, PATH_TYPE);
    if allow_bounds {
        opt_path_type_bounds_as_dyn_trait_type(p, path);
    }
}

/// This turns a parsed PATH_TYPE optionally into a DYN_TRAIT_TYPE
/// with a TYPE_BOUND_LIST
fn opt_path_type_bounds_as_dyn_trait_type(p: &mut Parser, path_type_marker: CompletedMarker) {
    if !p.at(T![+]) {
        return;
    }

    // First create a TYPE_BOUND from the completed PATH_TYPE
    let m = path_type_marker.precede(p).complete(p, TYPE_BOUND);

    // Next setup a marker for the TYPE_BOUND_LIST
    let m = m.precede(p);

    // This gets consumed here so it gets properly set
    // in the TYPE_BOUND_LIST
    p.eat(T![+]);

    // Parse rest of the bounds into the TYPE_BOUND_LIST
    let m = type_params::bounds_without_colon_m(p, m);

    // Finally precede everything with DYN_TRAIT_TYPE
    m.precede(p).complete(p, DYN_TRAIT_TYPE);
}
