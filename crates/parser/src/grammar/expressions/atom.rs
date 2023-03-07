use super::*;

// test expr_literals
// fn foo() {
//     let _ = true;
//     let _ = false;
//     let _ = 1;
//     let _ = 2.0;
//     let _ = b'a';
//     let _ = 'b';
//     let _ = "c";
//     let _ = r"d";
//     let _ = b"e";
//     let _ = br"f";
// }
pub(crate) const LITERAL_FIRST: TokenSet = TokenSet::new(&[
    T![true],
    T![false],
    INT_NUMBER,
    FLOAT_NUMBER,
    BYTE,
    CHAR,
    STRING,
    BYTE_STRING,
]);

pub(crate) fn literal(p: &mut Parser<'_>) -> Option<CompletedMarker> {
    if !p.at_ts(LITERAL_FIRST) {
        return None;
    }
    let m = p.start();
    p.bump_any();
    Some(m.complete(p, LITERAL))
}

// E.g. for after the break in `if break {}`, this should not match
pub(super) const ATOM_EXPR_FIRST: TokenSet =
    LITERAL_FIRST.union(paths::PATH_FIRST).union(TokenSet::new(&[
        T!['('],
        T!['{'],
        T!['['],
        T![|],
        T![async],
        T![box],
        T![break],
        T![const],
        T![continue],
        T![do],
        T![for],
        T![if],
        T![let],
        T![loop],
        T![match],
        T![move],
        T![return],
        T![static],
        T![try],
        T![unsafe],
        T![while],
        T![yield],
        LIFETIME_IDENT,
    ]));

pub(super) const EXPR_RECOVERY_SET: TokenSet = TokenSet::new(&[T![')'], T![']']]);

pub(super) fn atom_expr(
    p: &mut Parser<'_>,
    r: Restrictions,
) -> Option<(CompletedMarker, BlockLike)> {
    if let Some(m) = literal(p) {
        return Some((m, BlockLike::NotBlock));
    }
    if paths::is_path_start(p) {
        return Some(path_expr(p, r));
    }
    let la = p.nth(1);
    let done = match p.current() {
        T!['('] => tuple_expr(p),
        T!['['] => array_expr(p),
        T![if] => if_expr(p),
        T![let] => let_expr(p),
        T![_] => {
            // test destructuring_assignment_wildcard_pat
            // fn foo() {
            //     _ = 1;
            //     Some(_) = None;
            // }
            let m = p.start();
            p.bump(T![_]);
            m.complete(p, UNDERSCORE_EXPR)
        }
        T![loop] => loop_expr(p, None),
        T![box] => box_expr(p, None),
        T![while] => while_expr(p, None),
        T![try] => try_block_expr(p, None),
        T![match] => match_expr(p),
        T![return] => return_expr(p),
        T![yield] => yield_expr(p),
        T![do] if p.nth_at_contextual_kw(1, T![yeet]) => yeet_expr(p),
        T![continue] => continue_expr(p),
        T![break] => break_expr(p, r),

        LIFETIME_IDENT if la == T![:] => {
            let m = p.start();
            label(p);
            match p.current() {
                T![loop] => loop_expr(p, Some(m)),
                T![for] => for_expr(p, Some(m)),
                T![while] => while_expr(p, Some(m)),
                // test labeled_block
                // fn f() { 'label: {}; }
                T!['{'] => {
                    stmt_list(p);
                    m.complete(p, BLOCK_EXPR)
                }
                _ => {
                    // test_err misplaced_label_err
                    // fn main() {
                    //     'loop: impl
                    // }
                    p.error("expected a loop or block");
                    m.complete(p, ERROR);
                    return None;
                }
            }
        }
        // test effect_blocks
        // fn f() { unsafe { } }
        // fn f() { const { } }
        // fn f() { async { } }
        // fn f() { async move { } }
        T![const] | T![unsafe] | T![async] if la == T!['{'] => {
            let m = p.start();
            p.bump_any();
            stmt_list(p);
            m.complete(p, BLOCK_EXPR)
        }
        T![async] if la == T![move] && p.nth(2) == T!['{'] => {
            let m = p.start();
            p.bump(T![async]);
            p.eat(T![move]);
            stmt_list(p);
            m.complete(p, BLOCK_EXPR)
        }
        T!['{'] => {
            // test for_range_from
            // fn foo() {
            //    for x in 0 .. {
            //        break;
            //    }
            // }
            let m = p.start();
            stmt_list(p);
            m.complete(p, BLOCK_EXPR)
        }

        T![const] | T![static] | T![async] | T![move] | T![|] => closure_expr(p),
        T![for] if la == T![<] => closure_expr(p),
        T![for] => for_expr(p, None),

        _ => {
            p.err_and_bump("expected expression");
            return None;
        }
    };
    let blocklike =
        if BlockLike::is_blocklike(done.kind()) { BlockLike::Block } else { BlockLike::NotBlock };
    Some((done, blocklike))
}

// test tuple_expr
// fn foo() {
//     ();
//     (1);
//     (1,);
// }
fn tuple_expr(p: &mut Parser<'_>) -> CompletedMarker {
    assert!(p.at(T!['(']));
    let m = p.start();
    p.expect(T!['(']);

    let mut saw_comma = false;
    let mut saw_expr = false;
    while !p.at(EOF) && !p.at(T![')']) {
        saw_expr = true;

        // test tuple_attrs
        // const A: (i64, i64) = (1, #[cfg(test)] 2);
        if expr(p).is_none() {
            break;
        }

        if !p.at(T![')']) {
            saw_comma = true;
            p.expect(T![,]);
        }
    }
    p.expect(T![')']);
    m.complete(p, if saw_expr && !saw_comma { PAREN_EXPR } else { TUPLE_EXPR })
}

// test array_expr
// fn foo() {
//     [];
//     [1];
//     [1, 2,];
//     [1; 2];
// }
fn array_expr(p: &mut Parser<'_>) -> CompletedMarker {
    assert!(p.at(T!['[']));
    let m = p.start();

    let mut n_exprs = 0u32;
    let mut has_semi = false;

    p.bump(T!['[']);
    while !p.at(EOF) && !p.at(T![']']) {
        n_exprs += 1;

        // test array_attrs
        // const A: &[i64] = &[1, #[cfg(test)] 2];
        if expr(p).is_none() {
            break;
        }

        if n_exprs == 1 && p.eat(T![;]) {
            has_semi = true;
            continue;
        }

        if has_semi || !p.at(T![']']) && !p.expect(T![,]) {
            break;
        }
    }
    p.expect(T![']']);

    m.complete(p, ARRAY_EXPR)
}

// test lambda_expr
// fn foo() {
//     || ();
//     || -> i32 { 92 };
//     |x| x;
//     move |x: i32,| x;
//     async || {};
//     move || {};
//     async move || {};
//     static || {};
//     static move || {};
//     static async || {};
//     static async move || {};
//     for<'a> || {};
//     for<'a> move || {};
// }
fn closure_expr(p: &mut Parser<'_>) -> CompletedMarker {
    assert!(match p.current() {
        T![const] | T![static] | T![async] | T![move] | T![|] => true,
        T![for] => p.nth(1) == T![<],
        _ => false,
    });

    let m = p.start();

    if p.at(T![for]) {
        types::for_binder(p);
    }
    // test const_closure
    // fn main() { let cl = const || _ = 0; }
    p.eat(T![const]);
    p.eat(T![static]);
    p.eat(T![async]);
    p.eat(T![move]);

    if !p.at(T![|]) {
        p.error("expected `|`");
        return m.complete(p, CLOSURE_EXPR);
    }
    params::param_list_closure(p);
    if opt_ret_type(p) {
        // test lambda_ret_block
        // fn main() { || -> i32 { 92 }(); }
        block_expr(p);
    } else if p.at_ts(EXPR_FIRST) {
        // test closure_body_underscore_assignment
        // fn main() { || _ = 0; }
        expr(p);
    } else {
        p.error("expected expression");
    }
    m.complete(p, CLOSURE_EXPR)
}

// test if_expr
// fn foo() {
//     if true {};
//     if true {} else {};
//     if true {} else if false {} else {};
//     if S {};
//     if { true } { } else { };
// }
fn if_expr(p: &mut Parser<'_>) -> CompletedMarker {
    assert!(p.at(T![if]));
    let m = p.start();
    p.bump(T![if]);
    expr_no_struct(p);
    block_expr(p);
    if p.at(T![else]) {
        p.bump(T![else]);
        if p.at(T![if]) {
            if_expr(p);
        } else {
            block_expr(p);
        }
    }
    m.complete(p, IF_EXPR)
}

// test label
// fn foo() {
//     'a: loop {}
//     'b: while true {}
//     'c: for x in () {}
// }
fn label(p: &mut Parser<'_>) {
    assert!(p.at(LIFETIME_IDENT) && p.nth(1) == T![:]);
    let m = p.start();
    lifetime(p);
    p.bump_any();
    m.complete(p, LABEL);
}

// test loop_expr
// fn foo() {
//     loop {};
// }
fn loop_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
    assert!(p.at(T![loop]));
    let m = m.unwrap_or_else(|| p.start());
    p.bump(T![loop]);
    block_expr(p);
    m.complete(p, LOOP_EXPR)
}

// test while_expr
// fn foo() {
//     while true {};
//     while let Some(x) = it.next() {};
//     while { true } {};
// }
fn while_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
    assert!(p.at(T![while]));
    let m = m.unwrap_or_else(|| p.start());
    p.bump(T![while]);
    expr_no_struct(p);
    block_expr(p);
    m.complete(p, WHILE_EXPR)
}

// test for_expr
// fn foo() {
//     for x in [] {};
// }
fn for_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
    assert!(p.at(T![for]));
    let m = m.unwrap_or_else(|| p.start());
    p.bump(T![for]);
    patterns::pattern(p);
    p.expect(T![in]);
    expr_no_struct(p);
    block_expr(p);
    m.complete(p, FOR_EXPR)
}

// test let_expr
// fn foo() {
//     if let Some(_) = None && true {}
//     while 1 == 5 && (let None = None) {}
// }
fn let_expr(p: &mut Parser<'_>) -> CompletedMarker {
    let m = p.start();
    p.bump(T![let]);
    patterns::pattern_top(p);
    p.expect(T![=]);
    expr_let(p);
    m.complete(p, LET_EXPR)
}

// test match_expr
// fn foo() {
//     match () { };
//     match S {};
//     match { } { _ => () };
//     match { S {} } {};
// }
fn match_expr(p: &mut Parser<'_>) -> CompletedMarker {
    assert!(p.at(T![match]));
    let m = p.start();
    p.bump(T![match]);
    expr_no_struct(p);
    if p.at(T!['{']) {
        match_arm_list(p);
    } else {
        p.error("expected `{`");
    }
    m.complete(p, MATCH_EXPR)
}

pub(crate) fn match_arm_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.eat(T!['{']);

    // test match_arms_inner_attribute
    // fn foo() {
    //     match () {
    //         #![doc("Inner attribute")]
    //         #![doc("Can be")]
    //         #![doc("Stacked")]
    //         _ => (),
    //     }
    // }
    attributes::inner_attrs(p);

    while !p.at(EOF) && !p.at(T!['}']) {
        if p.at(T!['{']) {
            error_block(p, "expected match arm");
            continue;
        }
        match_arm(p);
    }
    p.expect(T!['}']);
    m.complete(p, MATCH_ARM_LIST);
}

// test match_arm
// fn foo() {
//     match () {
//         _ => (),
//         _ if Test > Test{field: 0} => (),
//         X | Y if Z => (),
//         | X | Y if Z => (),
//         | X => (),
//     };
// }
fn match_arm(p: &mut Parser<'_>) {
    let m = p.start();
    // test match_arms_outer_attributes
    // fn foo() {
    //     match () {
    //         #[cfg(feature = "some")]
    //         _ => (),
    //         #[cfg(feature = "other")]
    //         _ => (),
    //         #[cfg(feature = "many")]
    //         #[cfg(feature = "attributes")]
    //         #[cfg(feature = "before")]
    //         _ => (),
    //     }
    // }
    attributes::outer_attrs(p);

    patterns::pattern_top_r(p, TokenSet::EMPTY);
    if p.at(T![if]) {
        match_guard(p);
    }
    p.expect(T![=>]);
    let blocklike = match expr_stmt(p, None) {
        Some((_, blocklike)) => blocklike,
        None => BlockLike::NotBlock,
    };

    // test match_arms_commas
    // fn foo() {
    //     match () {
    //         _ => (),
    //         _ => {}
    //         _ => ()
    //     }
    // }
    if !p.eat(T![,]) && !blocklike.is_block() && !p.at(T!['}']) {
        p.error("expected `,`");
    }
    m.complete(p, MATCH_ARM);
}

// test match_guard
// fn foo() {
//     match () {
//         _ if foo => (),
//         _ if let foo = bar => (),
//     }
// }
fn match_guard(p: &mut Parser<'_>) -> CompletedMarker {
    assert!(p.at(T![if]));
    let m = p.start();
    p.bump(T![if]);
    expr(p);
    m.complete(p, MATCH_GUARD)
}

// test block
// fn a() {}
// fn b() { let _ = 1; }
// fn c() { 1; 2; }
// fn d() { 1; 2 }
pub(crate) fn block_expr(p: &mut Parser<'_>) {
    if !p.at(T!['{']) {
        p.error("expected a block");
        return;
    }
    let m = p.start();
    stmt_list(p);
    m.complete(p, BLOCK_EXPR);
}

fn stmt_list(p: &mut Parser<'_>) -> CompletedMarker {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    expr_block_contents(p);
    p.expect(T!['}']);
    m.complete(p, STMT_LIST)
}

// test return_expr
// fn foo() {
//     return;
//     return 92;
// }
fn return_expr(p: &mut Parser<'_>) -> CompletedMarker {
    assert!(p.at(T![return]));
    let m = p.start();
    p.bump(T![return]);
    if p.at_ts(EXPR_FIRST) {
        expr(p);
    }
    m.complete(p, RETURN_EXPR)
}

// test yield_expr
// fn foo() {
//     yield;
//     yield 1;
// }
fn yield_expr(p: &mut Parser<'_>) -> CompletedMarker {
    assert!(p.at(T![yield]));
    let m = p.start();
    p.bump(T![yield]);
    if p.at_ts(EXPR_FIRST) {
        expr(p);
    }
    m.complete(p, YIELD_EXPR)
}

// test yeet_expr
// fn foo() {
//     do yeet;
//     do yeet 1
// }
fn yeet_expr(p: &mut Parser<'_>) -> CompletedMarker {
    assert!(p.at(T![do]));
    assert!(p.nth_at_contextual_kw(1, T![yeet]));
    let m = p.start();
    p.bump(T![do]);
    p.bump_remap(T![yeet]);
    if p.at_ts(EXPR_FIRST) {
        expr(p);
    }
    m.complete(p, YEET_EXPR)
}

// test continue_expr
// fn foo() {
//     loop {
//         continue;
//         continue 'l;
//     }
// }
fn continue_expr(p: &mut Parser<'_>) -> CompletedMarker {
    assert!(p.at(T![continue]));
    let m = p.start();
    p.bump(T![continue]);
    if p.at(LIFETIME_IDENT) {
        lifetime(p);
    }
    m.complete(p, CONTINUE_EXPR)
}

// test break_expr
// fn foo() {
//     loop {
//         break;
//         break 'l;
//         break 92;
//         break 'l 92;
//     }
// }
fn break_expr(p: &mut Parser<'_>, r: Restrictions) -> CompletedMarker {
    assert!(p.at(T![break]));
    let m = p.start();
    p.bump(T![break]);
    if p.at(LIFETIME_IDENT) {
        lifetime(p);
    }
    // test break_ambiguity
    // fn foo(){
    //     if break {}
    //     while break {}
    //     for i in break {}
    //     match break {}
    // }
    if p.at_ts(EXPR_FIRST) && !(r.forbid_structs && p.at(T!['{'])) {
        expr(p);
    }
    m.complete(p, BREAK_EXPR)
}

// test try_block_expr
// fn foo() {
//     let _ = try {};
// }
fn try_block_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
    assert!(p.at(T![try]));
    let m = m.unwrap_or_else(|| p.start());
    // Special-case `try!` as macro.
    // This is a hack until we do proper edition support
    if p.nth_at(1, T![!]) {
        // test try_macro_fallback
        // fn foo() { try!(Ok(())); }
        let macro_call = p.start();
        let path = p.start();
        let path_segment = p.start();
        let name_ref = p.start();
        p.bump_remap(IDENT);
        name_ref.complete(p, NAME_REF);
        path_segment.complete(p, PATH_SEGMENT);
        path.complete(p, PATH);
        let _block_like = items::macro_call_after_excl(p);
        macro_call.complete(p, MACRO_CALL);
        return m.complete(p, MACRO_EXPR);
    }

    p.bump(T![try]);
    if p.at(T!['{']) {
        stmt_list(p);
    } else {
        p.error("expected a block");
    }
    m.complete(p, BLOCK_EXPR)
}

// test box_expr
// fn foo() {
//     let x = box 1i32;
//     let y = (box 1i32, box 2i32);
//     let z = Foo(box 1i32, box 2i32);
// }
fn box_expr(p: &mut Parser<'_>, m: Option<Marker>) -> CompletedMarker {
    assert!(p.at(T![box]));
    let m = m.unwrap_or_else(|| p.start());
    p.bump(T![box]);
    if p.at_ts(EXPR_FIRST) {
        expr(p);
    }
    m.complete(p, BOX_EXPR)
}
