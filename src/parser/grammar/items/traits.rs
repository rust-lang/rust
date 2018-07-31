use super::*;

pub(super) fn trait_item(p: &mut Parser) {
    assert!(p.at(TRAIT_KW));
    p.bump();
    name(p);
    p.expect(L_CURLY);
    p.expect(R_CURLY);
}

// test impl_item
// impl Foo {}
pub(super) fn impl_item(p: &mut Parser) {
    assert!(p.at(IMPL_KW));
    p.bump();
    if choose_type_params_over_qpath(p) {
        type_params::list(p);
    }

    // TODO: never type
    // impl ! {}

    // test impl_item_neg
    // impl !Send for X {}
    p.eat(EXCL);
    types::type_(p);
    if p.eat(FOR_KW) {
        types::type_(p);
    }
    type_params::where_clause(p);
    p.expect(L_CURLY);

    // test impl_item_items
    // impl F {
    //     type A = i32;
    //     const B: i32 = 92;
    //     fn foo() {}
    //     fn bar(&self) {}
    // }
    while !p.at(EOF) && !p.at(R_CURLY) {
        item(p);
    }
    p.expect(R_CURLY);
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
