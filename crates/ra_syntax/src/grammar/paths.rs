use super::*;

pub(super) const PATH_FIRST: TokenSet =
    token_set![IDENT, SELF_KW, SUPER_KW, CRATE_KW, COLONCOLON, L_ANGLE];

pub(super) fn is_path_start(p: &Parser) -> bool {
    match p.current() {
        IDENT | SELF_KW | SUPER_KW | CRATE_KW | COLONCOLON => true,
        _ => false,
    }
}

pub(super) fn use_path(p: &mut Parser) {
    path(p, Mode::Use)
}

pub(super) fn type_path(p: &mut Parser) {
    path(p, Mode::Type)
}

pub(super) fn expr_path(p: &mut Parser) {
    path(p, Mode::Expr)
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
    let mut qual = path.complete(p, PATH);
    loop {
        let use_tree = match p.nth(1) {
            STAR | L_CURLY => true,
            _ => false,
        };
        if p.at(COLONCOLON) && !use_tree {
            let path = qual.precede(p);
            p.bump();
            path_segment(p, mode, false);
            let path = path.complete(p, PATH);
            qual = path;
        } else {
            break;
        }
    }
}

fn path_segment(p: &mut Parser, mode: Mode, first: bool) {
    let m = p.start();
    // test qual_paths
    // type X = <A as B>::Output;
    // fn foo() { <usize as Default>::default(); }
    if first && p.eat(L_ANGLE) {
        types::type_(p);
        if p.eat(AS_KW) {
            if is_path_start(p) {
                types::path_type(p);
            } else {
                p.error("expected a trait");
            }
        }
        p.expect(R_ANGLE);
    } else {
        if first {
            p.eat(COLONCOLON);
        }
        match p.current() {
            IDENT => {
                name_ref(p);
                opt_path_type_args(p, mode);
            }
            // test crate_path
            // use crate::foo;
            SELF_KW | SUPER_KW | CRATE_KW => p.bump(),
            _ => {
                p.err_and_bump("expected identifier");
            }
        };
    }
    m.complete(p, PATH_SEGMENT);
}

fn opt_path_type_args(p: &mut Parser, mode: Mode) {
    match mode {
        Mode::Use => return,
        Mode::Type => {
            // test path_fn_trait_args
            // type F = Box<Fn(x: i32) -> ()>;
            if p.at(L_PAREN) {
                params::param_list_opt_patterns(p);
                opt_fn_ret_type(p);
            } else {
                type_args::opt_type_arg_list(p, false)
            }
        },
        Mode::Expr => type_args::opt_type_arg_list(p, true),
    }
}
