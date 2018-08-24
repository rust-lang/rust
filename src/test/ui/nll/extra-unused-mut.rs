// extra unused mut lint tests for #51918

// run-pass

#![feature(generators, nll)]
#![deny(unused_mut)]

fn ref_argument(ref _y: i32) {}

// #51801
fn mutable_upvar() {
    let mut x = 0;
    move || {
        x = 1;
    };
}

// #50897
fn generator_mutable_upvar() {
    let mut x = 0;
    move || {
        x = 1;
        yield;
    };
}

// #51830
fn ref_closure_argument() {
    let _ = Some(0).as_ref().map(|ref _a| true);
}

struct Expr {
    attrs: Vec<u32>,
}

// #51904
fn parse_dot_or_call_expr_with(mut attrs: Vec<u32>) {
    let x = Expr { attrs: vec![] };
    Some(Some(x)).map(|expr|
        expr.map(|mut expr| {
            attrs.push(666);
            expr.attrs = attrs;
            expr
        })
    );
}

// Found when trying to bootstrap rustc
fn if_guard(x: Result<i32, i32>) {
    match x {
        Ok(mut r) | Err(mut r) if true => r = 1,
        _ => (),
    }
}

fn main() {
    ref_argument(0);
    mutable_upvar();
    generator_mutable_upvar();
    ref_closure_argument();
    parse_dot_or_call_expr_with(Vec::new());
    if_guard(Ok(0));
}
