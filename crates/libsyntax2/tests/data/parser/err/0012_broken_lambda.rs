pub(super) fn process<'a, S: Sink<'a>>(builder: &mut S, tokens: &[Token], events: Vec<Event>) {
    let mut next_tok_idx = 0;
    let eat_ws = |idx: &mut usize, &mut | {
        while let Some(token) = tokens.get(*idx) {
            if !token.kind.is_trivia() {
                break;
            }
            builder.leaf(token.kind, token.len);
            *idx += 1
        }
    };
}
