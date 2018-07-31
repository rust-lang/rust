use super::*;

pub(super) fn use_item(p: &mut Parser) {
    assert!(p.at(USE_KW));
    p.bump();
    use_tree(p);
    p.expect(SEMI);
}

fn use_tree(p: &mut Parser) {
    let la = p.nth(1);
    let m = p.start();
    match (p.current(), la) {
        (STAR, _) => p.bump(),
        (COLONCOLON, STAR) => {
            p.bump();
            p.bump();
        }
        (L_CURLY, _) | (COLONCOLON, L_CURLY) => {
            if p.at(COLONCOLON) {
                p.bump();
            }
            nested_trees(p);
        }
        _ if paths::is_path_start(p) => {
            paths::use_path(p);
            match p.current() {
                AS_KW => {
                    alias(p);
                }
                COLONCOLON => {
                    p.bump();
                    match p.current() {
                        STAR => {
                            p.bump();
                        }
                        L_CURLY => nested_trees(p),
                        _ => {
                            // is this unreachable?
                            p.error("expected `{` or `*`");
                        }
                    }
                }
                _ => (),
            }
        }
        _ => {
            m.abandon(p);
            p.err_and_bump("expected one of `*`, `::`, `{`, `self`, `super`, `indent`");
            return;
        }
    }
    m.complete(p, USE_TREE);
}

fn nested_trees(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    p.bump();
    while !p.at(EOF) && !p.at(R_CURLY) {
        use_tree(p);
        if !p.at(R_CURLY) {
            p.expect(COMMA);
        }
    }
    p.expect(R_CURLY);
}
