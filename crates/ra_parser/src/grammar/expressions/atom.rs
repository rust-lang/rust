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
pub(crate) const LITERAL_FIRST: TokenSet = token_set![
    TRUE_KW,
    FALSE_KW,
    INT_NUMBER,
    FLOAT_NUMBER,
    BYTE,
    CHAR,
    STRING,
    RAW_STRING,
    BYTE_STRING,
    RAW_BYTE_STRING
];

pub(crate) fn literal(p: &mut Parser) -> Option<CompletedMarker> {
    if !p.at_ts(LITERAL_FIRST) {
        return None;
    }
    let m = p.start();
    p.bump();
    Some(m.complete(p, LITERAL))
}

// E.g. for after the break in `if break {}`, this should not match
pub(super) const ATOM_EXPR_FIRST: TokenSet =
    LITERAL_FIRST.union(paths::PATH_FIRST).union(token_set![
        L_PAREN,
        L_CURLY,
        L_BRACK,
        PIPE,
        MOVE_KW,
        BOX_KW,
        IF_KW,
        WHILE_KW,
        MATCH_KW,
        UNSAFE_KW,
        RETURN_KW,
        BREAK_KW,
        CONTINUE_KW,
        LIFETIME,
        ASYNC_KW,
        TRY_KW,
    ]);

const EXPR_RECOVERY_SET: TokenSet = token_set![LET_KW];

pub(super) fn atom_expr(p: &mut Parser, r: Restrictions) -> Option<(CompletedMarker, BlockLike)> {
    if let Some(m) = literal(p) {
        return Some((m, BlockLike::NotBlock));
    }
    if paths::is_path_start(p) || p.at(T![<]) {
        return Some(path_expr(p, r));
    }
    let la = p.nth(1);
    let done = match p.current() {
        T!['('] => tuple_expr(p),
        T!['['] => array_expr(p),
        T![|] => lambda_expr(p),
        T![move] if la == T![|] => lambda_expr(p),
        T![async] if la == T![|] || (la == T![move] && p.nth(2) == T![|]) => lambda_expr(p),
        T![if] => if_expr(p),

        T![loop] => loop_expr(p, None),
        T![box] => box_expr(p, None),
        T![for] => for_expr(p, None),
        T![while] => while_expr(p, None),
        T![try] => try_block_expr(p, None),
        LIFETIME if la == T![:] => {
            let m = p.start();
            label(p);
            match p.current() {
                T![loop] => loop_expr(p, Some(m)),
                T![for] => for_expr(p, Some(m)),
                T![while] => while_expr(p, Some(m)),
                T!['{'] => block_expr(p, Some(m)),
                _ => {
                    // test_err misplaced_label_err
                    // fn main() {
                    //     'loop: impl
                    // }
                    p.error("expected a loop");
                    m.complete(p, ERROR);
                    return None;
                }
            }
        }
        T![async] if la == T!['{'] || (la == T![move] && p.nth(2) == T!['{']) => {
            let m = p.start();
            p.bump();
            p.eat(T![move]);
            block_expr(p, Some(m))
        }
        T![match] => match_expr(p),
        T![unsafe] if la == T!['{'] => {
            let m = p.start();
            p.bump();
            block_expr(p, Some(m))
        }
        T!['{'] => block_expr(p, None),
        T![return] => return_expr(p),
        T![continue] => continue_expr(p),
        T![break] => break_expr(p, r),
        _ => {
            p.err_recover("expected expression", EXPR_RECOVERY_SET);
            return None;
        }
    };
    let blocklike = match done.kind() {
        IF_EXPR | WHILE_EXPR | FOR_EXPR | LOOP_EXPR | MATCH_EXPR | BLOCK_EXPR | TRY_BLOCK_EXPR => {
            BlockLike::Block
        }
        _ => BlockLike::NotBlock,
    };
    Some((done, blocklike))
}

// test tuple_expr
// fn foo() {
//     ();
//     (1);
//     (1,);
// }
fn tuple_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(T!['(']));
    let m = p.start();
    p.expect(T!['(']);

    let mut saw_comma = false;
    let mut saw_expr = false;
    while !p.at(EOF) && !p.at(T![')']) {
        saw_expr = true;
        if !p.at_ts(EXPR_FIRST) {
            p.error("expected expression");
            break;
        }
        expr(p);
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
fn array_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(T!['[']));
    let m = p.start();
    p.bump();
    if p.eat(T![']']) {
        return m.complete(p, ARRAY_EXPR);
    }

    // test first_array_member_attributes
    // pub const A: &[i64] = &[
    //    #[cfg(test)]
    //    1,
    //    2,
    // ];
    attributes::outer_attributes(p);

    expr(p);
    if p.eat(T![;]) {
        expr(p);
        p.expect(T![']']);
        return m.complete(p, ARRAY_EXPR);
    }
    while !p.at(EOF) && !p.at(T![']']) {
        p.expect(T![,]);
        if p.at(T![']']) {
            break;
        }

        // test subsequent_array_member_attributes
        // pub const A: &[i64] = &[
        //    1,
        //    #[cfg(test)]
        //    2,
        // ];
        attributes::outer_attributes(p);
        if !p.at_ts(EXPR_FIRST) {
            p.error("expected expression");
            break;
        }
        expr(p);
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
// }
fn lambda_expr(p: &mut Parser) -> CompletedMarker {
    assert!(
        p.at(T![|])
            || (p.at(T![move]) && p.nth(1) == T![|])
            || (p.at(T![async]) && p.nth(1) == T![|])
            || (p.at(T![async]) && p.nth(1) == T![move] && p.nth(2) == T![|])
    );
    let m = p.start();
    p.eat(T![async]);
    p.eat(T![move]);
    params::param_list_opt_types(p);
    if opt_fn_ret_type(p) {
        if !p.at(T!['{']) {
            p.error("expected `{`");
        }
    }
    expr(p);
    m.complete(p, LAMBDA_EXPR)
}

// test if_expr
// fn foo() {
//     if true {};
//     if true {} else {};
//     if true {} else if false {} else {};
//     if S {};
// }
fn if_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(T![if]));
    let m = p.start();
    p.bump();
    cond(p);
    block(p);
    if p.at(T![else]) {
        p.bump();
        if p.at(T![if]) {
            if_expr(p);
        } else {
            block(p);
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
fn label(p: &mut Parser) {
    assert!(p.at(LIFETIME) && p.nth(1) == T![:]);
    let m = p.start();
    p.bump();
    p.bump();
    m.complete(p, LABEL);
}

// test loop_expr
// fn foo() {
//     loop {};
// }
fn loop_expr(p: &mut Parser, m: Option<Marker>) -> CompletedMarker {
    assert!(p.at(T![loop]));
    let m = m.unwrap_or_else(|| p.start());
    p.bump();
    block(p);
    m.complete(p, LOOP_EXPR)
}

// test while_expr
// fn foo() {
//     while true {};
//     while let Some(x) = it.next() {};
// }
fn while_expr(p: &mut Parser, m: Option<Marker>) -> CompletedMarker {
    assert!(p.at(T![while]));
    let m = m.unwrap_or_else(|| p.start());
    p.bump();
    cond(p);
    block(p);
    m.complete(p, WHILE_EXPR)
}

// test for_expr
// fn foo() {
//     for x in [] {};
// }
fn for_expr(p: &mut Parser, m: Option<Marker>) -> CompletedMarker {
    assert!(p.at(T![for]));
    let m = m.unwrap_or_else(|| p.start());
    p.bump();
    patterns::pattern(p);
    p.expect(T![in]);
    expr_no_struct(p);
    block(p);
    m.complete(p, FOR_EXPR)
}

// test cond
// fn foo() { if let Some(_) = None {} }
// fn bar() {
//     if let Some(_) | Some(_) = None {}
//     if let | Some(_) = None {}
//     while let Some(_) | Some(_) = None {}
//     while let | Some(_) = None {}
// }
fn cond(p: &mut Parser) {
    let m = p.start();
    if p.eat(T![let]) {
        patterns::pattern_list(p);
        p.expect(T![=]);
    }
    expr_no_struct(p);
    m.complete(p, CONDITION);
}

// test match_expr
// fn foo() {
//     match () { };
//     match S {};
// }
fn match_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(T![match]));
    let m = p.start();
    p.bump();
    expr_no_struct(p);
    if p.at(T!['{']) {
        match_arm_list(p);
    } else {
        p.error("expected `{`")
    }
    m.complete(p, MATCH_EXPR)
}

pub(crate) fn match_arm_list(p: &mut Parser) {
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
    attributes::inner_attributes(p);

    while !p.at(EOF) && !p.at(T!['}']) {
        if p.at(T!['{']) {
            error_block(p, "expected match arm");
            continue;
        }

        // test match_arms_commas
        // fn foo() {
        //     match () {
        //         _ => (),
        //         _ => {}
        //         _ => ()
        //     }
        // }
        if match_arm(p).is_block() {
            p.eat(T![,]);
        } else if !p.at(T!['}']) {
            p.expect(T![,]);
        }
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
fn match_arm(p: &mut Parser) -> BlockLike {
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
    attributes::outer_attributes(p);

    patterns::pattern_list_r(p, TokenSet::empty());
    if p.at(T![if]) {
        match_guard(p);
    }
    p.expect(T![=>]);
    let blocklike = expr_stmt(p).1;
    m.complete(p, MATCH_ARM);
    blocklike
}

// test match_guard
// fn foo() {
//     match () {
//         _ if foo => (),
//     }
// }
fn match_guard(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(T![if]));
    let m = p.start();
    p.bump();
    expr(p);
    m.complete(p, MATCH_GUARD)
}

// test block_expr
// fn foo() {
//     {};
//     unsafe {};
//     'label: {};
// }
fn block_expr(p: &mut Parser, m: Option<Marker>) -> CompletedMarker {
    assert!(p.at(T!['{']));
    let m = m.unwrap_or_else(|| p.start());
    block(p);
    m.complete(p, BLOCK_EXPR)
}

// test return_expr
// fn foo() {
//     return;
//     return 92;
// }
fn return_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(T![return]));
    let m = p.start();
    p.bump();
    if p.at_ts(EXPR_FIRST) {
        expr(p);
    }
    m.complete(p, RETURN_EXPR)
}

// test continue_expr
// fn foo() {
//     loop {
//         continue;
//         continue 'l;
//     }
// }
fn continue_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(T![continue]));
    let m = p.start();
    p.bump();
    p.eat(LIFETIME);
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
fn break_expr(p: &mut Parser, r: Restrictions) -> CompletedMarker {
    assert!(p.at(T![break]));
    let m = p.start();
    p.bump();
    p.eat(LIFETIME);
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
fn try_block_expr(p: &mut Parser, m: Option<Marker>) -> CompletedMarker {
    assert!(p.at(T![try]));
    let m = m.unwrap_or_else(|| p.start());
    p.bump();
    block(p);
    m.complete(p, TRY_EXPR)
}

// test box_expr
// fn foo() {
//     let x = box 1i32;
//     let y = (box 1i32, box 2i32);
//     let z = Foo(box 1i32, box 2i32);
// }
fn box_expr(p: &mut Parser, m: Option<Marker>) -> CompletedMarker {
    assert!(p.at(T![box]));
    let m = m.unwrap_or_else(|| p.start());
    p.bump();
    if p.at_ts(EXPR_FIRST) {
        expr(p);
    }
    m.complete(p, BOX_EXPR)
}
