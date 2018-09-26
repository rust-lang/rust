use super::*;

// test trait_item
// trait T<U>: Hash + Clone where U: Copy {}
pub(super) fn trait_def(p: &mut Parser) {
    assert!(p.at(TRAIT_KW));
    p.bump();
    name_r(p, ITEM_RECOVERY_SET);
    type_params::opt_type_param_list(p);
    if p.at(COLON) {
        type_params::bounds(p);
    }
    type_params::opt_where_clause(p);
    if p.at(L_CURLY) {
        trait_item_list(p);
    } else {
        p.error("expected `{`");
    }
}

// test trait_item_list
// impl F {
//     type A: Clone;
//     const B: i32;
//     fn foo() {}
//     fn bar(&self);
// }
pub(crate) fn trait_item_list(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    let m = p.start();
    p.bump();
    while !p.at(EOF) && !p.at(R_CURLY) {
        if p.at(L_CURLY) {
            error_block(p, "expected an item");
            continue;
        }
        item_or_macro(p, true, ItemFlavor::Trait);
    }
    p.expect(R_CURLY);
    m.complete(p, ITEM_LIST);
}

// test impl_item
// impl Foo {}
pub(super) fn impl_item(p: &mut Parser) {
    assert!(p.at(IMPL_KW));
    p.bump();
    if choose_type_params_over_qpath(p) {
        type_params::opt_type_param_list(p);
    }

    // TODO: never type
    // impl ! {}

    // test impl_item_neg
    // impl !Send for X {}
    p.eat(EXCL);
    impl_type(p);
    if p.eat(FOR_KW) {
        impl_type(p);
    }
    type_params::opt_where_clause(p);
    if p.at(L_CURLY) {
        impl_item_list(p);
    } else {
        p.error("expected `{`");
    }
}

// test impl_item_list
// impl F {
//     type A = i32;
//     const B: i32 = 92;
//     fn foo() {}
//     fn bar(&self) {}
// }
pub(crate) fn impl_item_list(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    let m = p.start();
    p.bump();

    while !p.at(EOF) && !p.at(R_CURLY) {
        if p.at(L_CURLY) {
            error_block(p, "expected an item");
            continue;
        }
        item_or_macro(p, true, ItemFlavor::Mod);
    }
    p.expect(R_CURLY);
    m.complete(p, ITEM_LIST);
}

fn choose_type_params_over_qpath(p: &Parser) -> bool {
    // There's an ambiguity between generic parameters and qualified paths in impls.
    // If we see `<` it may start both, so we have to inspect some following tokens.
    // The following combinations can only start generics,
    // but not qualified paths (with one exception):
    //     `<` `>` - empty generic parameters
    //     `<` `#` - generic parameters with attributes
    //     `<` (LIFETIME|IDENT) `>` - single generic parameter
    //     `<` (LIFETIME|IDENT) `,` - first generic parameter in a list
    //     `<` (LIFETIME|IDENT) `:` - generic parameter with bounds
    //     `<` (LIFETIME|IDENT) `=` - generic parameter with a default
    // The only truly ambiguous case is
    //     `<` IDENT `>` `::` IDENT ...
    // we disambiguate it in favor of generics (`impl<T> ::absolute::Path<T> { ... }`)
    // because this is what almost always expected in practice, qualified paths in impls
    // (`impl <Type>::AssocTy { ... }`) aren't even allowed by type checker at the moment.
    if !p.at(L_ANGLE) {
        return false;
    }
    if p.nth(1) == POUND || p.nth(1) == R_ANGLE {
        return true;
    }
    (p.nth(1) == LIFETIME || p.nth(1) == IDENT)
        && (p.nth(2) == R_ANGLE || p.nth(2) == COMMA || p.nth(2) == COLON || p.nth(2) == EQ)
}

// test impl_type
// impl Type {}
// impl Trait1 for T {}
// impl impl NotType {}
// impl Trait2 for impl NotType {}
pub(crate) fn impl_type(p: &mut Parser) {
    if p.at(IMPL_KW) {
        p.error("expected trait or type");
        return;
    }
    types::type_(p);
}

