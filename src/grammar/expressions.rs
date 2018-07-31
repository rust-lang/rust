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
pub(super) fn literal(p: &mut Parser) -> Option<CompletedMarker> {
    match p.current() {
        TRUE_KW | FALSE_KW | INT_NUMBER | FLOAT_NUMBER | BYTE | CHAR | STRING | RAW_STRING
        | BYTE_STRING | RAW_BYTE_STRING => {
            let m = p.start();
            p.bump();
            Some(m.complete(p, LITERAL))
        }
        _ => None,
    }
}

pub(super) fn expr(p: &mut Parser) {
    let mut lhs = match prefix_expr(p) {
        Some(lhs) => lhs,
        None => return,
    };

    loop {
        lhs = match p.current() {
            L_PAREN => call_expr(p, lhs),
            DOT if p.nth(1) == IDENT => if p.nth(2) == L_PAREN {
                method_call_expr(p, lhs)
            } else {
                field_expr(p, lhs)
            },
            DOT if p.nth(1) == INT_NUMBER => field_expr(p, lhs),
            _ => break,
        }
    }
}

// test block
// fn a() {}
// fn b() { let _ = 1; }
// fn c() { 1; 2; }
// fn d() { 1; 2 }
pub(super) fn block(p: &mut Parser) {
    if !p.at(L_CURLY) {
        p.error("expected block");
    }
    let m = p.start();
    p.bump();
    while !p.at(EOF) && !p.at(R_CURLY) {
        match p.current() {
            LET_KW => let_stmt(p),
            c => {
                // test block_items
                // fn a() { fn b() {} }
                if items::ITEM_FIRST.contains(c) {
                    items::item(p)
                } else {
                    let expr_stmt = p.start();
                    expressions::expr(p);
                    if p.eat(SEMI) {
                        expr_stmt.complete(p, EXPR_STMT);
                    } else {
                        expr_stmt.abandon(p);
                    }
                }
            }
        }
    }
    p.expect(R_CURLY);
    m.complete(p, BLOCK);
}

// test let_stmt;
// fn foo() {
//     let a;
//     let b: i32;
//     let c = 92;
//     let d: i32 = 92;
// }
fn let_stmt(p: &mut Parser) {
    assert!(p.at(LET_KW));
    let m = p.start();
    p.bump();
    patterns::pattern(p);
    if p.at(COLON) {
        types::ascription(p);
    }
    if p.eat(EQ) {
        expressions::expr(p);
    }
    p.expect(SEMI);
    m.complete(p, LET_STMT);
}

fn prefix_expr(p: &mut Parser) -> Option<CompletedMarker> {
    let done = match p.current() {
        AMPERSAND => ref_expr(p),
        STAR => deref_expr(p),
        EXCL => not_expr(p),
        _ => return atom_expr(p),
    };
    Some(done)
}

// test ref_expr
// fn foo() {
//     let _ = &1;
//     let _ = &mut &f();
// }
fn ref_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(AMPERSAND));
    let m = p.start();
    p.bump();
    p.eat(MUT_KW);
    expr(p);
    m.complete(p, REF_EXPR)
}

// test deref_expr
// fn foo() {
//     **&1;
// }
fn deref_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(STAR));
    let m = p.start();
    p.bump();
    expr(p);
    m.complete(p, DEREF_EXPR)
}

// test not_expr
// fn foo() {
//     !!true;
// }
fn not_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(EXCL));
    let m = p.start();
    p.bump();
    expr(p);
    m.complete(p, NOT_EXPR)
}

fn atom_expr(p: &mut Parser) -> Option<CompletedMarker> {
    match literal(p) {
        Some(m) => return Some(m),
        None => (),
    }
    if paths::is_path_start(p) {
        return Some(path_expr(p));
    }

    let done = match p.current() {
        L_PAREN => tuple_expr(p),
        PIPE => lambda_expr(p),
        MOVE_KW if p.nth(1) == PIPE => lambda_expr(p),
        IF_KW => if_expr(p),
        _ => {
            p.err_and_bump("expected expression");
            return None;
        }
    };
    Some(done)
}

fn tuple_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(L_PAREN));
    let m = p.start();
    p.expect(L_PAREN);
    p.expect(R_PAREN);
    m.complete(p, TUPLE_EXPR)
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
    if fn_ret_type(p) {
        block(p);
    } else {
        expr(p)
    }
    m.complete(p, LAMBDA_EXPR)
}

// test if_expr
// fn foo() {
//     if true {};
//     if true {} else {};
//     if true {} else if false {} else {}
// }
fn if_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(IF_KW));
    let m = p.start();
    p.bump();
    expr(p);
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

// test call_expr
// fn foo() {
//     let _ = f();
//     let _ = f()(1)(1, 2,);
// }
fn call_expr(p: &mut Parser, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(L_PAREN));
    let m = lhs.precede(p);
    arg_list(p);
    m.complete(p, CALL_EXPR)
}

// test method_call_expr
// fn foo() {
//     x.foo();
//     y.bar(1, 2,);
// }
fn method_call_expr(p: &mut Parser, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(DOT) && p.nth(1) == IDENT && p.nth(2) == L_PAREN);
    let m = lhs.precede(p);
    p.bump();
    name_ref(p);
    arg_list(p);
    m.complete(p, METHOD_CALL_EXPR)
}

// test field_expr
// fn foo() {
//     x.foo;
//     x.0.bar;
// }
fn field_expr(p: &mut Parser, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(DOT) && (p.nth(1) == IDENT || p.nth(1) == INT_NUMBER));
    let m = lhs.precede(p);
    p.bump();
    if p.at(IDENT) {
        name_ref(p)
    } else {
        p.bump()
    }
    m.complete(p, FIELD_EXPR)
}

fn arg_list(p: &mut Parser) {
    assert!(p.at(L_PAREN));
    let m = p.start();
    p.bump();
    while !p.at(R_PAREN) && !p.at(EOF) {
        expr(p);
        if !p.at(R_PAREN) && !p.expect(COMMA) {
            break;
        }
    }
    p.eat(R_PAREN);
    m.complete(p, ARG_LIST);
}

// test path_expr
// fn foo() {
//     let _ = a;
//     let _ = a::b;
//     let _ = ::a::<b>;
// }
fn path_expr(p: &mut Parser) -> CompletedMarker {
    assert!(paths::is_path_start(p));
    let m = p.start();
    paths::expr_path(p);
    if p.at(L_CURLY) {
        struct_lit(p);
        m.complete(p, STRUCT_LIT)
    } else {
        m.complete(p, PATH_EXPR)
    }
}

// test struct_lit
// fn foo() {
//     S {};
//     S { x, y: 32, };
//     S { x, y: 32, ..Default::default() };
// }
fn struct_lit(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    p.bump();
    while !p.at(EOF) && !p.at(R_CURLY) {
        match p.current() {
            IDENT => {
                let m = p.start();
                name_ref(p);
                if p.eat(COLON) {
                    expr(p);
                }
                m.complete(p, STRUCT_LIT_FIELD);
            }
            DOTDOT => {
                p.bump();
                expr(p);
            }
            _ => p.err_and_bump("expected identifier"),
        }
        if !p.at(R_CURLY) {
            p.expect(COMMA);
        }
    }
    p.expect(R_CURLY);
}
