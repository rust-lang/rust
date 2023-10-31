// rustfmt-single_line_let_else_max_width: 100

fn main() {
    // Although this won't compile it still parses so make sure we can format empty else blocks
    let Some(x) = opt else {};

    // let-else may be formatted on a single line if they are "short"
    // and only contain a single expression
    let Some(x) = opt else { return };

    let Some(x) = opt else {
        return
    };

    let Some(x) = opt else { return; };

    let Some(x) = opt else {
        // nope
        return;
    };

    let Some(x) = opt else { let y = 1; return y };

    let Some(x) = y.foo("abc", fairly_long_identifier, "def", "123456", "string", "cheese") else { bar() };

    let Some(x) = abcdef().foo("abc", some_really_really_really_long_ident, "ident", "123456").bar().baz().qux("fffffffffffffffff") else { foo_bar() };
}

fn with_comments_around_else_keyword() {
    let Some(x) = opt /* pre else keyword block-comment */ else { return };

    let Some(x) = opt else /* post else keyword block-comment */ { return };

    let Some(x) = opt /* pre else keyword block-comment */ else /* post else keyword block-comment */ { return };

    let Some(x) = opt // pre else keyword line-comment
    else { return };

    let Some(x) = opt else
     // post else keyword line-comment
    { return };

    let Some(x) = opt // pre else keyword line-comment
    else
    // post else keyword line-comment
    { return };

}

fn unbreakable_initializer_expr_pre_formatting_let_else_length_near_max_width() {
    // Pre Formatting:
    // The length of `(indent)let pat = init else block;` is 100 (max_width)
    // Post Formatting:
    // The formatting is left unchanged!
    let Some(x) = some_really_really_really_really_really_really_really_long_name_A else { return };

    // Pre Formatting:
    // The length of `(indent)let pat = init else block;` is 100 (max_width)
    // Post Formatting:
    // The else keyword and opening brace remain on the same line as the initializer expr,
    // and the else block is formatted over multiple lines because we can't fit the
    // else block on the same line as the initializer expr.
    let Some(x) = some_really_really_really_really_really_really_really_long_name___B else {return};

    // Pre Formatting:
    // The length of `(indent)let pat = init else block;` is 100 (max_width)
    // Post Formatting:
    // The else keyword and opening brace remain on the same line as the initializer expr,
    // and the else block is formatted over multiple lines because we can't fit the
    // else block on the same line as the initializer expr.
    let Some(x) = some_really_really_really_really_long_name_____C else {some_divergent_function()};

    // Pre Formatting:
    // The length of `(indent)let pat = init else block;` is 101 (> max_width)
    // Post Formatting:
    // The else keyword and opening brace remain on the same line as the initializer expr,
    // and the else block is formatted over multiple lines because we can't fit the
    // else block on the same line as the initializer expr.
    let Some(x) = some_really_really_really_really_really_really_really_long_name__D else { return };
}

fn unbreakable_initializer_expr_pre_formatting_length_up_to_opening_brace_near_max_width() {
    // Pre Formatting:
    // The length of `(indent)let pat = init else {` is 99 (< max_width)
    // Post Formatting:
    // The else keyword and opening brace remain on the same line as the initializer expr,
    // and the else block is formatted over multiple lines because we can't fit the
    // else block on the same line as the initializer expr.
    let Some(x) = some_really_really_really_really_really_really_really_really_long_name___E else {return};

    // Pre Formatting:
    // The length of `(indent)let pat = init else {` is 101 (> max_width)
    // Post Formatting:
    // The else keyword and opening brace cannot fit on the same line as the initializer expr.
    // They are formatted on the next line.
    let Some(x) = some_really_really_really_really_really_really_really_really_long_name_____F else {return};
}

fn unbreakable_initializer_expr_pre_formatting_length_through_initializer_expr_near_max_width() {
    // Pre Formatting:
    // The length of `(indent)let pat = init` is 99 (< max_width)
    // Post Formatting:
    // The else keyword and opening brace cannot fit on the same line as the initializer expr.
    // They are formatted on the next line.
    let Some(x) = some_really_really_really_really_really_really_really_really_really_long_name___G else {return};

    // Pre Formatting:
    // The length of `(indent)let pat = init` is 100 (max_width)
    // Post Formatting:
    // Break after the `=` and put the initializer expr on it's own line.
    // Because the initializer expr is multi-lined the else is placed on it's own line.
    let Some(x) = some_really_really_really_really_really_really_really_really_really_long_name____H else {return};

    // Pre Formatting:
    // The length of `(indent)let pat = init` is 109 (> max_width)
    // Post Formatting:
    // Break after the `=` and put the initializer expr on it's own line.
    // Because the initializer expr is multi-lined the else is placed on it's own line.
    // The initializer expr has a length of 91, which when indented on the next line
    // The `(indent)init` line has a lengh of 99. This is the max length that the `init` can be
    // before we start running into max_width issues. I suspect this is becuase the shape is
    // accounting for the `;` at the end of the `let-else` statement.
    let Some(x) = some_really_really_really_really_really_really_really_really_really_really_long_name______I else {return};

    // Pre Formatting:
    // The length of `(indent)let pat = init` is 110 (> max_width)
    // Post Formatting:
    // Max length issues prevent us from formatting.
    // The initializer expr has a length of 92, which if it would be indented on the next line
    // the `(indent)init` line has a lengh of 100 which == max_width of 100.
    // One might expect formatting to succeed, but I suspect the reason we hit max_width issues is
    // because the Shape is accounting for the `;` at the end of the `let-else` statement.
    let Some(x) = some_really_really_really_really_really_really_really_really_really_really_really_long_nameJ else {return};
}

fn long_patterns() {
    let Foo {x: Bar(..), y: FooBar(..), z: Baz(..)} = opt else {
        return;
    };

    // with version=One we don't wrap long array patterns
    let [aaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbb, cccccccccccccccccc, dddddddddddddddddd] = opt else {
        return;
    };

    let ("aaaaaaaaaaaaaaaaaaa" | "bbbbbbbbbbbbbbbbb" | "cccccccccccccccccccccccc" | "dddddddddddddddd" | "eeeeeeeeeeeeeeee") = opt else {
        return;
    };

    let Some(Ok((Message::ChangeColor(super::color::Color::Rgb(r, g, b)), Point { x, y, z }))) = opt else {
        return;
    };
}

fn with_trailing_try_operator() {
    // Currently the trailing ? forces the else on the next line
    // This may be revisited in style edition 2024
    let Some(next_bucket) = ranking_rules[cur_ranking_rule_index].next_bucket(ctx, logger, &ranking_rule_universes[cur_ranking_rule_index])? else { return };

    // Maybe this is a workaround?
    let Ok(Some(next_bucket)) = ranking_rules[cur_ranking_rule_index].next_bucket(ctx, logger, &ranking_rule_universes[cur_ranking_rule_index]) else { return };
}
