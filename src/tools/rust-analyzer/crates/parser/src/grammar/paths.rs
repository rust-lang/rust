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

pub(super) fn vis_path(p: &mut Parser<'_>) {
    path(p, Mode::Vis);
}

pub(super) fn attr_path(p: &mut Parser<'_>) {
    path(p, Mode::Attr);
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
    Attr,
    Type,
    Expr,
    Vis,
}

fn path(p: &mut Parser<'_>, mode: Mode) -> Option<CompletedMarker> {
    let path = p.start();
    if path_segment(p, mode, true).is_none() {
        path.abandon(p);
        return None;
    }
    let qual = path.complete(p, PATH);
    Some(path_for_qualifier(p, mode, qual))
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
    expressions::EXPR_RECOVERY_SET.union(items::ITEM_RECOVERY_SET);
const TYPE_PATH_SEGMENT_RECOVERY_SET: TokenSet = types::TYPE_RECOVERY_SET;

fn path_segment(p: &mut Parser<'_>, mode: Mode, first: bool) -> Option<CompletedMarker> {
    let m = p.start();
    // test qual_paths
    // type X = <A as B>::Output;
    // fn foo() { <usize as Default>::default(); }
    if first && p.at(T![<]) {
        let m = p.start();
        p.bump(T![<]);
        // test_err angled_path_without_qual
        // type X = <()>;
        // type Y = <A as B>;
        types::type_(p);
        if p.eat(T![as]) {
            if is_use_path_start(p) {
                types::path_type_bounds(p, true);
            } else {
                p.error("expected a trait");
            }
        }
        p.expect(T![>]);
        m.complete(p, TYPE_ANCHOR);
        if !p.at(T![::]) {
            p.error("expected `::`");
        }
    } else {
        let mut empty = if first { !p.eat(T![::]) } else { true };
        if p.at_ts(PATH_NAME_REF_KINDS) {
            // test crate_path
            // use crate::foo;
            name_ref_mod_path(p);
            opt_path_args(p, mode);
        } else {
            let recover_set = match mode {
                Mode::Use => items::ITEM_RECOVERY_SET,
                Mode::Attr => {
                    items::ITEM_RECOVERY_SET.union(TokenSet::new(&[T![']'], T![=], T![#]]))
                }
                Mode::Vis => items::ITEM_RECOVERY_SET.union(TokenSet::new(&[T![')']])),
                Mode::Type => TYPE_PATH_SEGMENT_RECOVERY_SET,
                Mode::Expr => EXPR_PATH_SEGMENT_RECOVERY_SET,
            };
            empty &= p.err_recover(
                "expected identifier, `self`, `super`, `crate`, or `Self`",
                recover_set,
            );
            if empty {
                // test_err empty_segment
                // use crate::;
                m.abandon(p);
                return None;
            }
        }
    }
    Some(m.complete(p, PATH_SEGMENT))
}

pub(crate) fn opt_path_type_args(p: &mut Parser<'_>) {
    // test typepathfn_with_coloncolon
    // type F = Start::(Middle) -> (Middle)::End;
    // type GenericArg = S<Start(Middle)::End>;
    let m;
    if p.at(T![::]) && matches!(p.nth(2), T![<] | T!['(']) {
        m = p.start();
        p.bump(T![::]);
    } else if (p.current() == T![<] && p.nth(1) != T![=]) || p.current() == T!['('] {
        m = p.start();
    } else {
        return;
    }
    let current = p.current();
    if current == T![<] {
        // test_err generic_arg_list_recover
        // type T = T<0, ,T>;
        // type T = T::<0, ,T>;
        delimited(
            p,
            T![<],
            T![>],
            T![,],
            || "expected generic argument".into(),
            generic_args::GENERIC_ARG_FIRST,
            generic_args::generic_arg,
        );
        m.complete(p, GENERIC_ARG_LIST);
    } else if p.nth_at(1, T![..]) {
        // test return_type_syntax_in_path
        // fn foo<T>()
        // where
        //     T::method(..): Send,
        //     method(..): Send,
        //     method::(..): Send,
        // {}
        p.bump(T!['(']);
        p.bump(T![..]);
        p.expect(T![')']);
        m.complete(p, RETURN_TYPE_SYNTAX);
    } else {
        // test path_fn_trait_args
        // type F = Box<Fn(i32) -> ()>;
        // type F = Box<::Fn(i32) -> ()>;
        // type F = Box<Fn::(i32) -> ()>;
        // type F = Box<::Fn::(i32) -> ()>;
        delimited(
            p,
            T!['('],
            T![')'],
            T![,],
            || "expected type".into(),
            types::TYPE_FIRST,
            |p| {
                let progress = types::TYPE_FIRST.contains(p.current());
                generic_args::type_arg(p);
                progress
            },
        );
        m.complete(p, PARENTHESIZED_ARG_LIST);
        opt_ret_type(p);
    }
}

fn opt_path_args(p: &mut Parser<'_>, mode: Mode) {
    match mode {
        Mode::Use | Mode::Attr | Mode::Vis => {}
        Mode::Type => opt_path_type_args(p),
        Mode::Expr => generic_args::opt_generic_arg_list_expr(p),
    }
}
