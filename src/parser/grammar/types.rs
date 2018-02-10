use super::*;

pub(super) fn ty(p: &mut Parser) {
    match p.current() {
        L_PAREN => paren_or_tuple_ty(p),
        IDENT => path_type(p),
        _ => {
            p.error("expected type");
        }
    }
}

fn paren_or_tuple_ty(p: &mut Parser) {
    assert!(p.at(L_PAREN));
    let m = p.start();
    p.bump();
    let mut n_types: u32 = 0;
    let mut trailing_comma: bool = false;
    while !p.at(EOF) && !p.at(R_PAREN) {
        n_types += 1;
        ty(p);
        if p.eat(COMMA) {
            trailing_comma = true;
        } else {
            trailing_comma = false;
            break;
        }
    }
    p.expect(R_PAREN);

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

fn path_type(p: &mut Parser) {
    assert!(p.at(IDENT));
    let m = p.start();
    paths::type_path(p);
    m.complete(p, PATH_TYPE);
}
