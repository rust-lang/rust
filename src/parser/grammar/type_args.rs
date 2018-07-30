use super::*;

pub(super) fn list(p: &mut Parser, colon_colon_required: bool) {
    let m;
    match (colon_colon_required, p.nth(0), p.nth(1)) {
        (_, COLONCOLON, L_ANGLE) => {
            m = p.start();
            p.bump();
            p.bump();
        }
        (false, L_ANGLE, _) => {
            m = p.start();
            p.bump();
        }
        _ => return
    };

    while !p.at(EOF) && !p.at(R_ANGLE) {
        types::type_(p);
        if !p.at(R_ANGLE) && !p.expect(COMMA) {
            break;
        }
    }
    p.expect(R_ANGLE);
    m.complete(p, TYPE_ARG_LIST);
}
