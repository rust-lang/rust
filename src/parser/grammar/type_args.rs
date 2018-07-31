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
        _ => return,
    };

    while !p.at(EOF) && !p.at(R_ANGLE) {
        type_arg(p);
        if !p.at(R_ANGLE) && !p.expect(COMMA) {
            break;
        }
    }
    p.expect(R_ANGLE);
    m.complete(p, TYPE_ARG_LIST);
}

// test type_arg
// type A = B<'static, i32, Item=u64>
fn type_arg(p: &mut Parser) {
    let m = p.start();
    match p.current() {
        LIFETIME => {
            p.bump();
            m.complete(p, LIFETIME_ARG);
        },
        IDENT if p.nth(1) == EQ => {
            name_ref(p);
            p.bump();
            types::type_(p);
            m.complete(p, ASSOC_TYPE_ARG);
        },
        _ => {
            types::type_(p);
            m.complete(p, TYPE_ARG);
        },
    }
}
