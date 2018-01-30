use super::*;

pub(super) fn is_path_start(p: &Parser) -> bool {
    AnyOf(&[IDENT, SELF_KW, SUPER_KW, COLONCOLON]).is_ahead(p)
}

pub(super) fn use_path(p: &mut Parser) {
    path(p)
}

pub(super) fn type_path(p: &mut Parser) {
    path(p)
}

fn path(p: &mut Parser) {
    if !is_path_start(p) {
        return;
    }
    let path = p.start();
    path_segment(p, true);
    let mut qual = path.complete(p, PATH);
    loop {
        if p.at(COLONCOLON) && !items::is_use_tree_start(p.nth(1)) {
            let path = qual.precede(p);
            p.bump();
            path_segment(p, false);
            let path = path.complete(p, PATH);
            qual = path;
        } else {
            break;
        }
    }
}

fn path_segment(p: &mut Parser, first: bool) {
    let segment = p.start();
    if first {
        p.eat(COLONCOLON);
    }
    match p.current() {
        IDENT | SELF_KW | SUPER_KW => p.bump(),
        _ => p.error().message("expected identifier").emit(),
    };
    segment.complete(p, PATH_SEGMENT);
}
