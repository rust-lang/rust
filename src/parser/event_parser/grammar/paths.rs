use super::*;

pub (crate) fn is_path_start(p: &Parser) -> bool {
    AnyOf(&[IDENT, SELF_KW, SUPER_KW, COLONCOLON]).is_ahead(p)
}

pub(crate) fn use_path(p: &mut Parser) {
    if !is_path_start(p) {
        return;
    }
    let mut prev = p.mark();
    node(p, PATH, |p| {
        path_segment(p, true);
    });
    many(p, |p| {
        let curr = p.mark();
        if p.current() == COLONCOLON && !items::is_use_tree_start(p.raw_lookahead(1)) {
            node(p, PATH, |p| {
                p.bump();
                path_segment(p, false);
                p.forward_parent(prev, curr);
                prev = curr;
            });
            true
        } else {
            false
        }
    });
}

fn path_segment(p: &mut Parser, first: bool) {
    node(p, PATH_SEGMENT, |p| {
        if first {
            p.eat(COLONCOLON);
        }
        match p.current() {
            IDENT | SELF_KW | SUPER_KW => {
                p.bump();
            },
            _ => {
                p.error()
                    .message("expected identifier")
                    .emit();
            }
        };
    })
}