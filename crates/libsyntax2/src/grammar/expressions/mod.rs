mod atom;

use super::*;
pub(super) use self::atom::literal;

const EXPR_FIRST: TokenSet = LHS_FIRST;

pub(super) fn expr(p: &mut Parser) -> BlockLike {
    let r = Restrictions { forbid_structs: false, prefer_stmt: false };
    expr_bp(p, r, 1)
}

pub(super) fn expr_stmt(p: &mut Parser) -> BlockLike {
    let r = Restrictions { forbid_structs: false, prefer_stmt: true };
    expr_bp(p, r, 1)
}

fn expr_no_struct(p: &mut Parser) {
    let r = Restrictions { forbid_structs: true, prefer_stmt: false };
    expr_bp(p, r, 1);
}

// test block
// fn a() {}
// fn b() { let _ = 1; }
// fn c() { 1; 2; }
// fn d() { 1; 2 }
pub(super) fn block(p: &mut Parser) {
    if !p.at(L_CURLY) {
        p.error("expected block");
        return;
    }
    atom::block_expr(p);
}

#[derive(Clone, Copy)]
struct Restrictions {
    forbid_structs: bool,
    prefer_stmt: bool,
}

enum Op {
    Simple,
    Composite(SyntaxKind, u8),
}

fn current_op(p: &Parser) -> (u8, Op) {
    if p.at_compound2(PLUS, EQ) {
        return (1, Op::Composite(PLUSEQ, 2));
    }
    if p.at_compound2(MINUS, EQ) {
        return (1, Op::Composite(MINUSEQ, 2));
    }
    if p.at_compound3(L_ANGLE, L_ANGLE, EQ) {
        return (1, Op::Composite(SHLEQ, 3));
    }
    if p.at_compound3(R_ANGLE, R_ANGLE, EQ) {
        return (1, Op::Composite(SHREQ, 3));
    }
    if p.at_compound2(PIPE, PIPE) {
        return (3, Op::Composite(PIPEPIPE, 2));
    }
    if p.at_compound2(AMP, AMP) {
        return (4, Op::Composite(AMPAMP, 2));
    }
    if p.at_compound2(L_ANGLE, EQ) {
        return (5, Op::Composite(LTEQ, 2));
    }
    if p.at_compound2(R_ANGLE, EQ) {
        return (5, Op::Composite(GTEQ, 2));
    }
    if p.at_compound2(L_ANGLE, L_ANGLE) {
        return (9, Op::Composite(SHL, 2));
    }
    if p.at_compound2(R_ANGLE, R_ANGLE) {
        return (9, Op::Composite(SHR, 2));
    }

    let bp = match p.current() {
        EQ => 1,
        DOTDOT => 2,
        EQEQ | NEQ | L_ANGLE | R_ANGLE => 5,
        PIPE => 6,
        CARET => 7,
        AMP => 8,
        MINUS | PLUS => 10,
        STAR | SLASH | PERCENT => 11,
        _ => 0,
    };
    (bp, Op::Simple)
}

// Parses expression with binding power of at least bp.
fn expr_bp(p: &mut Parser, r: Restrictions, bp: u8) -> BlockLike {
    let mut lhs = match lhs(p, r) {
        Some(lhs) => {
            // test stmt_bin_expr_ambiguity
            // fn foo() {
            //     let _ = {1} & 2;
            //     {1} &2;
            // }
            if r.prefer_stmt && is_block(lhs.kind()) {
                return BlockLike::Block;
            }
            lhs
        }
        None => return BlockLike::NotBlock,
    };

    loop {
        let is_range = p.current() == DOTDOT;
        let (op_bp, op) = current_op(p);
        if op_bp < bp {
            break;
        }
        let m = lhs.precede(p);
        match op {
            Op::Simple => p.bump(),
            Op::Composite(kind, n) => {
                p.bump_compound(kind, n);
            }
        }
        expr_bp(p, r, op_bp + 1);
        lhs = m.complete(p, if is_range { RANGE_EXPR } else { BIN_EXPR });
    }
    BlockLike::NotBlock
}

// test no_semi_after_block
// fn foo() {
//     if true {}
//     loop {}
//     match () {}
//     while true {}
//     for _ in () {}
//     {}
//     {}
// }
fn is_block(kind: SyntaxKind) -> bool {
    match kind {
        IF_EXPR | WHILE_EXPR | FOR_EXPR | LOOP_EXPR | MATCH_EXPR | BLOCK_EXPR => true,
        _ => false,
    }
}

const LHS_FIRST: TokenSet =
    token_set_union![
        token_set![AMP, STAR, EXCL, DOTDOT, MINUS],
        atom::ATOM_EXPR_FIRST,
    ];

fn lhs(p: &mut Parser, r: Restrictions) -> Option<CompletedMarker> {
    let m;
    let kind = match p.current() {
        // test ref_expr
        // fn foo() {
        //     let _ = &1;
        //     let _ = &mut &f();
        // }
        AMP => {
            m = p.start();
            p.bump();
            p.eat(MUT_KW);
            REF_EXPR
        }
        // test unary_expr
        // fn foo() {
        //     **&1;
        //     !!true;
        //     --1;
        // }
        STAR | EXCL | MINUS => {
            m = p.start();
            p.bump();
            PREFIX_EXPR
        }
        // test full_range_expr
        // fn foo() { xs[..]; }
        DOTDOT => {
            m = p.start();
            p.bump();
            if EXPR_FIRST.contains(p.current()) {
                expr_bp(p, r, 2);
            }
            return Some(m.complete(p, RANGE_EXPR));
        }
        _ => {
            let lhs = atom::atom_expr(p, r)?;
            return Some(postfix_expr(p, r, lhs));
        }
    };
    expr_bp(p, r, 255);
    Some(m.complete(p, kind))
}

fn postfix_expr(p: &mut Parser, r: Restrictions, mut lhs: CompletedMarker) -> CompletedMarker {
    let mut allow_calls = !r.prefer_stmt || !is_block(lhs.kind());
    loop {
        lhs = match p.current() {
            // test stmt_postfix_expr_ambiguity
            // fn foo() {
            //     match () {
            //         _ => {}
            //         () => {}
            //         [] => {}
            //     }
            // }
            L_PAREN if allow_calls => call_expr(p, lhs),
            L_BRACK if allow_calls => index_expr(p, lhs),
            DOT if p.nth(1) == IDENT => if p.nth(2) == L_PAREN || p.nth(2) == COLONCOLON {
                method_call_expr(p, lhs)
            } else {
                field_expr(p, lhs)
            },
            DOT if p.nth(1) == INT_NUMBER => field_expr(p, lhs),
            // test postfix_range
            // fn foo() { let x = 1..; }
            DOTDOT if !EXPR_FIRST.contains(p.nth(1)) => {
                let m = lhs.precede(p);
                p.bump();
                m.complete(p, RANGE_EXPR)
            }
            QUESTION => try_expr(p, lhs),
            AS_KW => cast_expr(p, lhs),
            _ => break,
        };
        allow_calls = true
    }
    lhs
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

// test index_expr
// fn foo() {
//     x[1][2];
// }
fn index_expr(p: &mut Parser, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(L_BRACK));
    let m = lhs.precede(p);
    p.bump();
    expr(p);
    p.expect(R_BRACK);
    m.complete(p, INDEX_EXPR)
}

// test method_call_expr
// fn foo() {
//     x.foo();
//     y.bar::<T>(1, 2,);
// }
fn method_call_expr(p: &mut Parser, lhs: CompletedMarker) -> CompletedMarker {
    assert!(
        p.at(DOT) && p.nth(1) == IDENT
            && (p.nth(2) == L_PAREN || p.nth(2) == COLONCOLON)
    );
    let m = lhs.precede(p);
    p.bump();
    name_ref(p);
    type_args::type_arg_list(p, true);
    if p.at(L_PAREN) {
        arg_list(p);
    }
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

// test try_expr
// fn foo() {
//     x?;
// }
fn try_expr(p: &mut Parser, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(QUESTION));
    let m = lhs.precede(p);
    p.bump();
    m.complete(p, TRY_EXPR)
}

// test cast_expr
// fn foo() {
//     82 as i32;
// }
fn cast_expr(p: &mut Parser, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(AS_KW));
    let m = lhs.precede(p);
    p.bump();
    types::type_(p);
    m.complete(p, CAST_EXPR)
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
//     let _ = format!();
// }
fn path_expr(p: &mut Parser, r: Restrictions) -> CompletedMarker {
    assert!(paths::is_path_start(p) || p.at(L_ANGLE));
    let m = p.start();
    paths::expr_path(p);
    match p.current() {
        L_CURLY if !r.forbid_structs => {
            struct_lit(p);
            m.complete(p, STRUCT_LIT)
        }
        EXCL => {
            items::macro_call_after_excl(p);
            m.complete(p, MACRO_CALL)
        }
        _ => m.complete(p, PATH_EXPR)
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
