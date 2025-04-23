//@ compile-flags: -Zunpretty=expanded
//@ edition:2024
//@ check-pass

// This test covers the AST pretty-printer's insertion of parentheses in some
// macro metavariable edge cases. Synthetic parentheses (i.e. not appearing in
// the syntax tree) need to be printed in order for the printed code to be valid
// Rust syntax. We also test negative cases: the pretty-printer should not be
// synthesizing parentheses indiscriminately; only where necessary.

#![feature(if_let_guard)]

macro_rules! expr {
    ($expr:expr) => { $expr };
}

macro_rules! stmt {
    ($stmt:stmt) => { $stmt };
}

fn break_labeled_loop() {
    let no_paren = 'outer: loop {
        break 'outer expr!('inner: loop { break 'inner 1; } + 1);
    };

    let paren_around_break_value = 'outer: loop {
        break expr!('inner: loop { break 'inner 1; } + 1);
    };

    macro_rules! breaking {
        ($value:expr) => {
            break $value
        };
    }

    let paren_around_break_value = loop {
        breaking!('inner: loop { break 'inner 1; } + 1);
    };
}

fn if_let() {
    macro_rules! if_let {
        ($pat:pat, $expr:expr) => {
            if let $pat = $expr {}
        };
    }

    if let no_paren = true && false {}
    if_let!(paren_around_binary, true && false);
    if_let!(no_paren, true);

    struct Struct {}
    match () {
        _ if let no_paren = Struct {} => {}
    }
}

fn let_else() {
    let no_paren = expr!(1 + 1) else { return; };
    let paren_around_loop = expr!(loop {}) else { return; };
}

fn local() {
    macro_rules! let_expr_minus_one {
        ($pat:pat, $expr:expr) => {
            let $pat = $expr - 1;
        };
    }

    let void;
    let_expr_minus_one!(no_paren, match void {});

    macro_rules! let_expr_else_return {
        ($pat:pat, $expr:expr) => {
            let $pat = $expr else { return; };
        };
    }

    let_expr_else_return!(no_paren, void());
}

fn match_arm() {
    macro_rules! match_arm {
        ($pat:pat, $expr:expr) => {
            match () { $pat => $expr }
        };
    }

    match_arm!(no_paren, 1 - 1);
    match_arm!(paren_around_brace, { 1 } - 1);
}

/// https://github.com/rust-lang/rust/issues/98790
fn stmt_boundary() {
    macro_rules! expr_as_stmt {
        ($expr:expr) => {
            stmt!($expr)
        };
    }

    let paren_around_match;
    expr_as_stmt!(match paren_around_match {} | true);

    macro_rules! minus_one {
        ($expr:expr) => {
            expr_as_stmt!($expr - 1)
        };
    }

    let (no_paren, paren_around_loop);
    minus_one!(no_paren);
    minus_one!(match paren_around_match {});
    minus_one!(match paren_around_match {}());
    minus_one!(match paren_around_match {}[0]);
    minus_one!(loop { break paren_around_loop; });
}

fn vis_inherited() {
    macro_rules! vis_inherited {
        ($vis:vis struct) => {
            $vis struct Struct;
        };
    }

    vis_inherited!(struct);
}
