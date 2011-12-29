// xfail-test

// FIXME: Parser doesn't distinguish expression in parentheses (as in
// this example) from one that is not!  It is somewhat of a pain to
// fix this though there are no theoretical difficulties.  We could
// either add paren to the AST (better for pretty-print, I suppose) or
// modify the parser to track whether the expression in question is
// parenthesized.  I did the latter once and it was a bit of pain but
// not terribly difficult.  We could also the decision as to whether
// something is an "expression with a value" down into the
// parse_expr() codepath, where we *know* if there are parentheses or
// not, but we'd probably have to be a bit more careful then with
// clearing the top-level restrction flag (which we ought to do
// anyhow!)

fn main() {
    let v = [1f, 2f, 3f];
    let w =
        if true { (vec::any(v) { |e| float::nonnegative(e) }) }
        else { false };
    assert w;
}

