// rustfmt-style_edition: 2027

fn main() {
    _ = if let Some(term_node) = sema
        .token_ancestors_with_macros(token.clone())
        .find(|node| {
            matches!(
                node.kind(),
                BLOCK_EXPR | ARG_LIST | PAREN_EXPR | ARRAY_EXPR | MATCH_EXPR
            )
        }) {
        match term_node.kind() {}
    };
}
