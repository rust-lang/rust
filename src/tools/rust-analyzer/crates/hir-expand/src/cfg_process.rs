//! Processes out #[cfg] and #[cfg_attr] attributes from the input for the derive macro
use std::{cell::OnceCell, ops::ControlFlow};

use ::tt::TextRange;
use base_db::Crate;
use cfg::CfgExpr;
use parser::T;
use smallvec::SmallVec;
use syntax::{
    AstNode, PreorderWithTokens, SyntaxElement, SyntaxNode, SyntaxToken, WalkEvent,
    ast::{self, HasAttrs, TokenTreeChildren},
};
use syntax_bridge::DocCommentDesugarMode;

use crate::{
    attrs::{AttrId, Meta, expand_cfg_attr, is_item_tree_filtered_attr},
    db::ExpandDatabase,
    fixup::{self, SyntaxFixupUndoInfo},
    span_map::SpanMapRef,
    tt::{self, DelimSpan, Span},
};

struct ItemIsCfgedOut;

#[derive(Debug)]
struct ExpandedAttrToProcess {
    range: TextRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NextExpandedAttrState {
    NotStarted,
    InTheMiddle,
}

#[derive(Debug)]
struct AstAttrToProcess {
    range: TextRange,
    expanded_attrs: SmallVec<[ExpandedAttrToProcess; 1]>,
    expanded_attrs_idx: usize,
    next_expanded_attr: NextExpandedAttrState,
    pound_span: Span,
    brackets_span: DelimSpan,
    /// If `Some`, this is an inner attribute.
    excl_span: Option<Span>,
}

fn macro_input_callback(
    db: &dyn ExpandDatabase,
    is_derive: bool,
    censor_item_tree_attr_ids: &[AttrId],
    krate: Crate,
    default_span: Span,
    span_map: SpanMapRef<'_>,
) -> impl FnMut(&mut PreorderWithTokens, &WalkEvent<SyntaxElement>) -> (bool, Vec<tt::Leaf>) {
    let cfg_options = OnceCell::new();
    let cfg_options = move || *cfg_options.get_or_init(|| krate.cfg_options(db));

    let mut should_strip_attr = {
        let mut item_tree_attr_id = 0;
        let mut censor_item_tree_attr_ids_index = 0;
        move || {
            let mut result = false;
            if let Some(&next_censor_attr_id) =
                censor_item_tree_attr_ids.get(censor_item_tree_attr_ids_index)
                && next_censor_attr_id.item_tree_index() == item_tree_attr_id
            {
                censor_item_tree_attr_ids_index += 1;
                result = true;
            }
            item_tree_attr_id += 1;
            result
        }
    };

    let mut attrs = Vec::new();
    let mut attrs_idx = 0;
    let mut has_inner_attrs_owner = false;
    let mut in_attr = false;
    let mut done_with_attrs = false;
    let mut did_top_attrs = false;
    move |preorder, event| {
        match event {
            WalkEvent::Enter(SyntaxElement::Node(node)) => {
                if done_with_attrs {
                    return (true, Vec::new());
                }

                if ast::Attr::can_cast(node.kind()) {
                    in_attr = true;
                    let node_range = node.text_range();
                    while attrs
                        .get(attrs_idx)
                        .is_some_and(|it: &AstAttrToProcess| it.range != node_range)
                    {
                        attrs_idx += 1;
                    }
                } else if !in_attr && let Some(has_attrs) = ast::AnyHasAttrs::cast(node.clone()) {
                    // Attributes of the form `key = value` have `ast::Expr` in them, which returns `Some` for
                    // `AnyHasAttrs::cast()`, so we also need to check `in_attr`.

                    if has_inner_attrs_owner {
                        has_inner_attrs_owner = false;
                        return (true, Vec::new());
                    }

                    if did_top_attrs && !is_derive {
                        // Derives need all attributes handled, but attribute macros need only the top attributes handled.
                        done_with_attrs = true;
                        return (true, Vec::new());
                    }
                    did_top_attrs = true;

                    if let Some(inner_attrs_node) = has_attrs.inner_attributes_node()
                        && inner_attrs_node != *node
                    {
                        has_inner_attrs_owner = true;
                    }

                    let node_attrs = ast::attrs_including_inner(&has_attrs);

                    attrs.clear();
                    node_attrs.clone().for_each(|attr| {
                        let span_for = |token: Option<SyntaxToken>| {
                            token
                                .map(|token| span_map.span_for_range(token.text_range()))
                                .unwrap_or(default_span)
                        };
                        attrs.push(AstAttrToProcess {
                            range: attr.syntax().text_range(),
                            pound_span: span_for(attr.pound_token()),
                            brackets_span: DelimSpan {
                                open: span_for(attr.l_brack_token()),
                                close: span_for(attr.r_brack_token()),
                            },
                            excl_span: attr
                                .excl_token()
                                .map(|token| span_map.span_for_range(token.text_range())),
                            expanded_attrs: SmallVec::new(),
                            expanded_attrs_idx: 0,
                            next_expanded_attr: NextExpandedAttrState::NotStarted,
                        });
                    });

                    attrs_idx = 0;
                    let strip_current_item = expand_cfg_attr(
                        node_attrs,
                        &cfg_options,
                        |attr, _container, range, top_attr| {
                            // Find the attr.
                            while attrs[attrs_idx].range != top_attr.syntax().text_range() {
                                attrs_idx += 1;
                            }

                            let mut strip_current_attr = false;
                            match attr {
                                Meta::NamedKeyValue { name, .. } => {
                                    if name
                                        .is_none_or(|name| !is_item_tree_filtered_attr(name.text()))
                                    {
                                        strip_current_attr = should_strip_attr();
                                    }
                                }
                                Meta::TokenTree { path, tt } => {
                                    if path.is1("cfg") {
                                        let cfg_expr = CfgExpr::parse_from_ast(
                                            &mut TokenTreeChildren::new(&tt).peekable(),
                                        );
                                        if cfg_options().check(&cfg_expr) == Some(false) {
                                            return ControlFlow::Break(ItemIsCfgedOut);
                                        }
                                        strip_current_attr = true;
                                    } else if path.segments.len() != 1
                                        || !is_item_tree_filtered_attr(path.segments[0].text())
                                    {
                                        strip_current_attr = should_strip_attr();
                                    }
                                }
                                Meta::Path { path } => {
                                    if path.segments.len() != 1
                                        || !is_item_tree_filtered_attr(path.segments[0].text())
                                    {
                                        strip_current_attr = should_strip_attr();
                                    }
                                }
                            }

                            if !strip_current_attr {
                                attrs[attrs_idx]
                                    .expanded_attrs
                                    .push(ExpandedAttrToProcess { range });
                            }

                            ControlFlow::Continue(())
                        },
                    );
                    attrs_idx = 0;

                    if strip_current_item.is_some() {
                        preorder.skip_subtree();
                        attrs.clear();

                        'eat_comma: {
                            // If there is a comma after this node, eat it too.
                            let mut events_until_comma = 0;
                            for event in preorder.clone() {
                                match event {
                                    WalkEvent::Enter(SyntaxElement::Node(_))
                                    | WalkEvent::Leave(_) => {}
                                    WalkEvent::Enter(SyntaxElement::Token(token)) => {
                                        let kind = token.kind();
                                        if kind == T![,] {
                                            break;
                                        } else if !kind.is_trivia() {
                                            break 'eat_comma;
                                        }
                                    }
                                }
                                events_until_comma += 1;
                            }
                            preorder.nth(events_until_comma);
                        }

                        return (false, Vec::new());
                    }
                }
            }
            WalkEvent::Leave(SyntaxElement::Node(node)) => {
                if ast::Attr::can_cast(node.kind()) {
                    in_attr = false;
                    attrs_idx += 1;
                }
            }
            WalkEvent::Enter(SyntaxElement::Token(token)) => {
                if !in_attr {
                    return (true, Vec::new());
                }

                let Some(ast_attr) = attrs.get_mut(attrs_idx) else {
                    return (true, Vec::new());
                };
                let token_range = token.text_range();
                let Some(expanded_attr) = ast_attr.expanded_attrs.get(ast_attr.expanded_attrs_idx)
                else {
                    // No expanded attributes in this `ast::Attr`, or we finished them all already, either way
                    // the remaining tokens should be discarded.
                    return (false, Vec::new());
                };
                match ast_attr.next_expanded_attr {
                    NextExpandedAttrState::NotStarted => {
                        if token_range.start() >= expanded_attr.range.start() {
                            // We started the next attribute.
                            let mut insert_tokens = Vec::with_capacity(3);
                            insert_tokens.push(tt::Leaf::Punct(tt::Punct {
                                char: '#',
                                spacing: tt::Spacing::Alone,
                                span: ast_attr.pound_span,
                            }));
                            if let Some(span) = ast_attr.excl_span {
                                insert_tokens.push(tt::Leaf::Punct(tt::Punct {
                                    char: '!',
                                    spacing: tt::Spacing::Alone,
                                    span,
                                }));
                            }
                            insert_tokens.push(tt::Leaf::Punct(tt::Punct {
                                char: '[',
                                spacing: tt::Spacing::Alone,
                                span: ast_attr.brackets_span.open,
                            }));

                            ast_attr.next_expanded_attr = NextExpandedAttrState::InTheMiddle;

                            return (true, insert_tokens);
                        } else {
                            // Before any attribute or between the attributes.
                            return (false, Vec::new());
                        }
                    }
                    NextExpandedAttrState::InTheMiddle => {
                        if token_range.start() >= expanded_attr.range.end() {
                            // Finished the current attribute.
                            let insert_tokens = vec![tt::Leaf::Punct(tt::Punct {
                                char: ']',
                                spacing: tt::Spacing::Alone,
                                span: ast_attr.brackets_span.close,
                            })];

                            ast_attr.next_expanded_attr = NextExpandedAttrState::NotStarted;
                            ast_attr.expanded_attrs_idx += 1;

                            // It's safe to ignore the current token because between attributes
                            // there is always at least one token we skip - either the closing bracket
                            // in `#[]` or the comma in case of multiple attrs in `cfg_attr` expansion.
                            return (false, insert_tokens);
                        } else {
                            // Still in the middle.
                            return (true, Vec::new());
                        }
                    }
                }
            }
            WalkEvent::Leave(SyntaxElement::Token(_)) => {}
        }
        (true, Vec::new())
    }
}

pub(crate) fn attr_macro_input_to_token_tree(
    db: &dyn ExpandDatabase,
    node: &SyntaxNode,
    span_map: SpanMapRef<'_>,
    span: Span,
    is_derive: bool,
    censor_item_tree_attr_ids: &[AttrId],
    krate: Crate,
) -> (tt::TopSubtree, SyntaxFixupUndoInfo) {
    let fixups = fixup::fixup_syntax(span_map, node, span, DocCommentDesugarMode::ProcMacro);
    (
        syntax_bridge::syntax_node_to_token_tree_modified(
            node,
            span_map,
            fixups.append,
            fixups.remove,
            span,
            DocCommentDesugarMode::ProcMacro,
            macro_input_callback(db, is_derive, censor_item_tree_attr_ids, krate, span, span_map),
        ),
        fixups.undo_info,
    )
}

pub fn check_cfg_attr_value(
    db: &dyn ExpandDatabase,
    attr: &ast::TokenTree,
    krate: Crate,
) -> Option<bool> {
    let cfg_expr = CfgExpr::parse_from_ast(&mut TokenTreeChildren::new(attr).peekable());
    krate.cfg_options(db).check(&cfg_expr)
}
