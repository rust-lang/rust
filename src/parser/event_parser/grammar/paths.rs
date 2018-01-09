use super::*;

pub(crate) fn use_path(p: &mut Parser) {
    if !AnyOf(&[IDENT, COLONCOLON]).is_ahead(p) {
        return;
    }
    node(p, PATH, |p| {
        p.eat(COLONCOLON);
        path_segment(p);
    })
}

fn path_segment(p: &mut Parser) -> bool {
    node_if(p, IDENT, PATH_SEGMENT, |p| ())
}