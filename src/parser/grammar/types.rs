use super::*;

pub(super) fn type_ref(p: &mut Parser) {
    match p.current() {
        IDENT => p.bump(),
        L_PAREN => {
            p.bump();
            p.expect(R_PAREN);
        }
        _ => {
            p.error("expected type");
        }
    }
}
