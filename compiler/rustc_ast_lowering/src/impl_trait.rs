/// Returns `true` if the parent path contains impl trait syntax:
/// For example given `impl Bla<impl Foo>`, this function would
/// return true for the path for `impl Foo`
pub(crate) fn parent_contains_impl_trait(cx: &LoweringContext<'_>, path: &ast::Path) -> bool {
    let ast::Path { span: path_span, segments, tokens: _ } = path;

    if let Some(parent_path_span) = path_span.parent_callsite() {
        return matches!(cx.source_map().span_to_snippet(parent_path_span), Ok(s) if s.starts_with("impl "));
    }

    // This can be from a parameter list:
    // like in `fn foo(a: impl Bla<impl Foo<T>..`) somewhere
    // in a block or other nested context.
    let parent_node = cx.source_map().span_to_enclosing_node(*path_span).next();

    if let Some(node) = parent_node {
        let content_str = cx.source_map().span_to_snippet(node.span).unwrap_or_default();
        let segments_strs =
            segments.iter().map(|s| cx.source_map().span_to_snippet(s.span()).unwrap_or_default());

        let path_str = segments_strs.collect::<Vec<_>>().join("::");
        // Check if parent contains "impl Trait", except for the current path:
        let impl_trait_pattern = format!("impl {}", path_str);
        if content_str.contains("impl") && content_str.contains(&impl_trait_pattern) {
            return true;
        }
    }

    false
}
