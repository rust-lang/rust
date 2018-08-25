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
pub(crate) const LITERAL_FIRST: TokenSet =
    token_set![TRUE_KW, FALSE_KW, INT_NUMBER, FLOAT_NUMBER, BYTE, CHAR,
               STRING, RAW_STRING, BYTE_STRING, RAW_BYTE_STRING];

pub(crate) fn literal(p: &mut Parser) -> Option<CompletedMarker> {
    if !LITERAL_FIRST.contains(p.current()) {
        return None;
    }
    let m = p.start();
    p.bump();
    Some(m.complete(p, LITERAL))
}

pub(super) const ATOM_EXPR_FIRST: TokenSet =
    token_set_union![
        LITERAL_FIRST,
        token_set![L_PAREN, PIPE, MOVE_KW, IF_KW, WHILE_KW, MATCH_KW, UNSAFE_KW, L_CURLY, RETURN_KW,
                   IDENT, SELF_KW, SUPER_KW, COLONCOLON, BREAK_KW, CONTINUE_KW, LIFETIME ],
    ];

pub(super) fn atom_expr(p: &mut Parser, r: Restrictions) -> Option<CompletedMarker> {
    match literal(p) {
        Some(m) => return Some(m),
        None => (),
    }
    if paths::is_path_start(p) || p.at(L_ANGLE) {
        return Some(path_expr(p, r));
    }
    let la = p.nth(1);
    let done = match p.current() {
        L_PAREN => tuple_expr(p),
        L_BRACK => array_expr(p),
        PIPE => lambda_expr(p),
        MOVE_KW if la == PIPE => lambda_expr(p),
        IF_KW => if_expr(p),

        LOOP_KW => loop_expr(p, None),
        FOR_KW => for_expr(p, None),
        WHILE_KW => while_expr(p, None),
        LIFETIME if la == COLON => {
            let m = p.start();
            label(p);
            match p.current() {
                LOOP_KW => loop_expr(p, Some(m)),
                FOR_KW => for_expr(p, Some(m)),
                WHILE_KW => while_expr(p, Some(m)),
                _ => {
                    p.error("expected a loop");
                    return None;
                }
            }
        }

        MATCH_KW => match_expr(p),
        UNSAFE_KW if la == L_CURLY => block_expr(p),
        L_CURLY => block_expr(p),
        RETURN_KW => return_expr(p),
        CONTINUE_KW => continue_expr(p),
        BREAK_KW => break_expr(p),
        _ => {
            p.err_and_bump("expected expression");
            return None;
        }
    };
    Some(done)
}

// test tuple_expr
// fn foo() {
//     ();
//     (1);
//     (1,);
// }
fn tuple_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(L_PAREN));
    let m = p.start();
    p.expect(L_PAREN);

    let mut saw_comma = false;
    let mut saw_expr = false;
    while !p.at(EOF) && !p.at(R_PAREN) {
        saw_expr = true;
        expr(p);
        if !p.at(R_PAREN) {
            saw_comma = true;
            p.expect(COMMA);
        }
    }
    p.expect(R_PAREN);
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
    assert!(p.at(L_BRACK));
    let m = p.start();
    p.bump();
    if p.eat(R_BRACK) {
        return m.complete(p, ARRAY_EXPR);
    }
    expr(p);
    if p.eat(SEMI) {
        expr(p);
        p.expect(R_BRACK);
        return m.complete(p, ARRAY_EXPR);
    }
    while !p.at(EOF) && !p.at(R_BRACK) {
        p.expect(COMMA);
        if !p.at(R_BRACK) {
            expr(p);
        }
    }
    p.expect(R_BRACK);
    m.complete(p, ARRAY_EXPR)
}

// test lambda_expr
// fn foo() {
//     || ();
//     || -> i32 { 92 };
//     |x| x;
//     move |x: i32,| x;
// }
fn lambda_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(PIPE) || (p.at(MOVE_KW) && p.nth(1) == PIPE));
    let m = p.start();
    p.eat(MOVE_KW);
    params::param_list_opt_types(p);
    if opt_fn_ret_type(p) {
        block(p);
    } else {
        expr(p);
    }
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
    assert!(p.at(IF_KW));
    let m = p.start();
    p.bump();
    cond(p);
    block(p);
    if p.at(ELSE_KW) {
        p.bump();
        if p.at(IF_KW) {
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
    assert!(p.at(LIFETIME) && p.nth(1) == COLON);
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
    assert!(p.at(LOOP_KW));
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
    assert!(p.at(WHILE_KW));
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
    assert!(p.at(FOR_KW));
    let m = m.unwrap_or_else(|| p.start());
    p.bump();
    patterns::pattern(p);
    p.expect(IN_KW);
    expr_no_struct(p);
    block(p);
    m.complete(p, FOR_EXPR)
}

// test cond
// fn foo() { if let Some(_) = None {} }
fn cond(p: &mut Parser) {
    if p.eat(LET_KW) {
        patterns::pattern(p);
        p.expect(EQ);
    }
    expr_no_struct(p)
}

// test match_expr
// fn foo() {
//     match () { };
//     match S {};
// }
fn match_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(MATCH_KW));
    let m = p.start();
    p.bump();
    expr_no_struct(p);
    if p.at(L_CURLY) {
        match_arm_list(p);
    } else {
        p.error("expected `{`")
    }
    m.complete(p, MATCH_EXPR)
}

fn match_arm_list(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    let m = p.start();
    p.eat(L_CURLY);
    while !p.at(EOF) && !p.at(R_CURLY) {
        // test match_arms_commas
        // fn foo() {
        //     match () {
        //         _ => (),
        //         _ => {}
        //         _ => ()
        //     }
        // }
        if match_arm(p).is_block() {
            p.eat(COMMA);
        } else if !p.at(R_CURLY) {
            p.expect(COMMA);
        }
    }
    p.expect(R_CURLY);
    m.complete(p, MATCH_ARM_LIST);
}

// test match_arm
// fn foo() {
//     match () {
//         _ => (),
//         X | Y if Z => (),
//     };
// }
fn match_arm(p: &mut Parser) -> BlockLike {
    let m = p.start();
    loop {
        patterns::pattern(p);
        if !p.eat(PIPE) {
            break;
        }
    }
    if p.eat(IF_KW) {
        expr_no_struct(p);
    }
    p.expect(FAT_ARROW);
    let ret = expr_stmt(p);
    m.complete(p, MATCH_ARM);
    ret
}

// test block_expr
// fn foo() {
//     {};
//     unsafe {};
// }
pub(super) fn block_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(L_CURLY) || p.at(UNSAFE_KW) && p.nth(1) == L_CURLY);
    let m = p.start();
    p.eat(UNSAFE_KW);
    block(p);
    m.complete(p, BLOCK_EXPR)
}

// test return_expr
// fn foo() {
//     return;
//     return 92;
// }
fn return_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(RETURN_KW));
    let m = p.start();
    p.bump();
    if EXPR_FIRST.contains(p.current()) {
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
    assert!(p.at(CONTINUE_KW));
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
fn break_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(BREAK_KW));
    let m = p.start();
    p.bump();
    p.eat(LIFETIME);
    if EXPR_FIRST.contains(p.current()) {
        expr(p);
    }
    m.complete(p, BREAK_EXPR)
}
