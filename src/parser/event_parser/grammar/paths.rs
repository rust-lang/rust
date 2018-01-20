use super::*;

pub(crate) fn is_path_start(p: &Parser) -> bool {
    AnyOf(&[IDENT, SELF_KW, SUPER_KW, COLONCOLON]).is_ahead(p)
}

pub(crate) fn use_path(p: &mut Parser) {
    if !is_path_start(p) {
        return;
    }
    let mut prev = p.mark();
    p.start(PATH);
    path_segment(p, true);
    p.finish();
    loop {
        let curr = p.mark();
        if p.at(COLONCOLON) && !items::is_use_tree_start(p.raw_lookahead(1)) {
            p.start(PATH);
            p.bump();
            path_segment(p, false);
            p.forward_parent(prev, curr);
            prev = curr;
            p.finish();
        } else {
            break;
        }
    }
}

fn path_segment(p: &mut Parser, first: bool) {
    p.start(PATH_SEGMENT);
    if first {
        p.eat(COLONCOLON);
    }
    match p.current() {
        IDENT | SELF_KW | SUPER_KW => {
            p.bump();
        }
        _ => {
            p.error()
                .message("expected identifier")
                .emit();
        }
    };
    p.finish();
}
