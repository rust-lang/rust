use super::*;

pub(super) const PATH_FIRST: TokenSet =
    TokenSet::new(&[IDENT, T![self], T![super], T![crate], T![Self], T![:], T![<]]);

pub(super) fn is_path_start(p: &Parser<'_>) -> bool {
    is_use_path_start(p) || p.at(T![<]) || p.at(T![Self])
}

pub(super) fn is_use_path_start(p: &Parser<'_>) -> bool {
    match p.current() {
        IDENT | T![self] | T![super] | T![crate] => true,
        T![:] if p.at(T![::]) => true,
        _ => false,
    }
}

pub(super) fn use_path(p: &mut Parser<'_>) {
    path(p, Mode::Use);
}

pub(crate) fn type_path(p: &mut Parser<'_>) {
    path(p, Mode::Type);
}

pub(super) fn expr_path(p: &mut Parser<'_>) {
    path(p, Mode::Expr);
}

pub(crate) fn type_path_for_qualifier(
    p: &mut Parser<'_>,
    qual: CompletedMarker,
) -> CompletedMarker {
    path_for_qualifier(p, Mode::Type, qual)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Mode {
    Use,
    Type,
    Expr,
}

fn path(p: &mut Parser<'_>, mode: Mode) {
    let path = p.start();
    path_segment(p, mode, true);
    let qual = path.complete(p, PATH);
    path_for_qualifier(p, mode, qual);
}

fn path_for_qualifier(
    p: &mut Parser<'_>,
    mode: Mode,
    mut qual: CompletedMarker,
) -> CompletedMarker {
    loop {
        let use_tree = mode == Mode::Use && matches!(p.nth(2), T![*] | T!['{']);
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

const EXPR_PATH_SEGMENT_RECOVERY_SET: TokenSet =
    items::ITEM_RECOVERY_SET.union(TokenSet::new(&[T![')'], T![,], T![let]]));
const TYPE_PATH_SEGMENT_RECOVERY_SET: TokenSet = types::TYPE_RECOVERY_SET;

fn path_segment(p: &mut Parser<'_>, mode: Mode, first: bool) {
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
        let empty = if first {
            p.eat(T![::]);
            false
        } else {
            true
        };
        match p.current() {
            IDENT => {
                name_ref(p);
                opt_path_type_args(p, mode);
            }
            // test crate_path
            // use crate::foo;
            T![self] | T![super] | T![crate] | T![Self] => {
                let m = p.start();
                p.bump_any();
                m.complete(p, NAME_REF);
            }
            _ => {
                let recover_set = match mode {
                    Mode::Use => items::ITEM_RECOVERY_SET,
                    Mode::Type => TYPE_PATH_SEGMENT_RECOVERY_SET,
                    Mode::Expr => EXPR_PATH_SEGMENT_RECOVERY_SET,
                };
                p.err_recover("expected identifier", recover_set);
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

fn opt_path_type_args(p: &mut Parser<'_>, mode: Mode) {
    match mode {
        Mode::Use => {}
        Mode::Type => {
            // test typepathfn_with_coloncolon
            // type F = Start::(Middle) -> (Middle)::End;
            if p.at(T![::]) && p.nth_at(2, T!['(']) {
                p.bump(T![::]);
            }
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
