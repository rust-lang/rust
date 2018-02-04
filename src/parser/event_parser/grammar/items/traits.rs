use super::*;

pub(super) fn trait_item(p: &mut Parser) {
    assert!(p.at(TRAIT_KW));
    p.bump();
    p.expect(IDENT);
    p.expect(L_CURLY);
    p.expect(R_CURLY);
}

pub(super) fn impl_item(p: &mut Parser) {
    assert!(p.at(IMPL_KW));
    p.bump();
    p.expect(IDENT);
    p.expect(L_CURLY);
    p.expect(R_CURLY);
}
