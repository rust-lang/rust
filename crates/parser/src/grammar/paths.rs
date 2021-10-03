use super::*;

pub(super) const PATH_FIRST: TokenSet =
    TokenSet::new(&[IDENT, T![self], T![super], T![crate], T![:], T![<]]);

pub(super) fn is_path_start(p: &Parser) -> bool {
    is_use_path_start(p) || p.at(T![<])
}

pub(super) fn is_use_path_start(p: &Parser) -> bool {
    match p.current() {
        IDENT | T![self] | T![super] | T![crate] => true,
        T![:] if p.at(T![::]) => true,
        _ => false,
    }
}

pub(super) fn use_path(p: &mut Parser) {
    path(p, Mode::Use);
}

pub(crate) fn type_path(p: &mut Parser) {
    path(p, Mode::Type);
}

pub(super) fn expr_path(p: &mut Parser) {
    path(p, Mode::Expr);
}

pub(crate) fn type_path_for_qualifier(p: &mut Parser, qual: CompletedMarker) -> CompletedMarker {
    path_for_qualifier(p, Mode::Type, qual)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Mode {
    Use,
    Type,
    Expr,
}

fn path(p: &mut Parser, mode: Mode) {
    let path = p.start();
    path_segment(p, mode, true);
    let qual = path.complete(p, PATH);
    path_for_qualifier(p, mode, qual);
}

fn path_for_qualifier(p: &mut Parser, mode: Mode, mut qual: CompletedMarker) -> CompletedMarker {
    loop {
        let use_tree = matches!(p.nth(2), T![*] | T!['{']);
        if p.at(T![::]) && !use_tree {
            let path = qual.precede(p);
            p.bump(T![::]);
            path_segment(p, mode, false);
            let path = path.complete(p, PATH);
            qual = path;
        } else {
            return qual;
        }
    }
}

fn path_segment(p: &mut Parser, mode: Mode, first: bool) {
    let m = p.start();
    // test qual_paths
    // type X = <A as B>::Output;
    // fn foo() { <usize as Default>::default(); }
    if first && p.eat(T![<]) {
        types::type_(p);
        if p.eat(T![as]) {
            if is_use_path_start(p) {
                types::path_type(p);
            } else {
                p.error("expected a trait");
            }
        }
        p.expect(T![>]);
    } else {
        let mut empty = true;
        if first {
            p.eat(T![::]);
            empty = false;
        }
        match p.current() {
            IDENT => {
                name_ref(p);
                opt_path_type_args(p, mode);
            }
            // test crate_path
            // use crate::foo;
            T![self] | T![super] | T![crate] => {
                let m = p.start();
                p.bump_any();
                m.complete(p, NAME_REF);
            }
            _ => {
                p.err_recover("expected identifier", items::ITEM_RECOVERY_SET);
                if empty {
                    // test_err empty_segment
                    // use crate::;
                    m.abandon(p);
                    return;
                }
            }
        };
    }
    m.complete(p, PATH_SEGMENT);
}

fn opt_path_type_args(p: &mut Parser, mode: Mode) {
    match mode {
        Mode::Use => {}
        Mode::Type => {
            // test path_fn_trait_args
            // type F = Box<Fn(i32) -> ()>;
            if p.at(T!['(']) {
                params::param_list_fn_trait(p);
                opt_ret_type(p);
            } else {
                generic_args::opt_generic_arg_list(p, false);
            }
        }
        Mode::Expr => generic_args::opt_generic_arg_list(p, true),
    }
}
