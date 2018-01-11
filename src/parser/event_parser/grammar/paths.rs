use super::*;

pub(crate) fn use_path(p: &mut Parser) {
    if !AnyOf(&[IDENT, COLONCOLON]).is_ahead(p) {
        return;
    }
    node(p, PATH, |p| {
        path_segment(p, true);
    });
    many(p, |p| {
        node_if(p, COLONCOLON, PATH, |p| {
            path_segment(p, false);
        })
    });
}

fn path_segment(p: &mut Parser, first: bool) {
    node(p, PATH_SEGMENT, |p| {
        if first {
            p.eat(COLONCOLON);
        }
        p.expect(IDENT);
    })
}