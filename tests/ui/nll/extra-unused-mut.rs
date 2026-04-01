// extra unused mut lint tests for #51918

//@ check-pass

#![feature(coroutines)]
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
fn coroutine_mutable_upvar() {
    let mut x = 0;

    #[coroutine]
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
    Some(Some(x)).map(|expr| {
        expr.map(|mut expr| {
            attrs.push(666);
            expr.attrs = attrs;
            expr
        })
    });
}

// Found when trying to bootstrap rustc
fn if_guard(x: Result<i32, i32>) {
    match x {
        Ok(mut r) | Err(mut r) if true => r = 1,
        _ => (),
    }
}

// #59620
fn nested_closures() {
    let mut i = 0;
    [].iter().for_each(|_: &i32| {
        [].iter().for_each(move |_: &i32| {
            i += 1;
        });
    });
}

fn main() {}
