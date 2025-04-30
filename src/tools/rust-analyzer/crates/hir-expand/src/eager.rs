//! Eager expansion related utils
//!
//! Here is a dump of a discussion from Vadim Petrochenkov about Eager Expansion and
//! Its name resolution :
//!
//! > Eagerly expanded macros (and also macros eagerly expanded by eagerly expanded macros,
//! > which actually happens in practice too!) are resolved at the location of the "root" macro
//! > that performs the eager expansion on its arguments.
//! > If some name cannot be resolved at the eager expansion time it's considered unresolved,
//! > even if becomes available later (e.g. from a glob import or other macro).
//!
//! > Eagerly expanded macros don't add anything to the module structure of the crate and
//! > don't build any speculative module structures, i.e. they are expanded in a "flat"
//! > way even if tokens in them look like modules.
//!
//! > In other words, it kinda works for simple cases for which it was originally intended,
//! > and we need to live with it because it's available on stable and widely relied upon.
//!
//!
//! See the full discussion : <https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler/topic/Eager.20expansion.20of.20built-in.20macros>
use base_db::Crate;
use span::SyntaxContext;
use syntax::{AstPtr, Parse, SyntaxElement, SyntaxNode, TextSize, WalkEvent, ted};
use syntax_bridge::DocCommentDesugarMode;
use triomphe::Arc;

use crate::{
    AstId, EagerCallInfo, ExpandError, ExpandResult, ExpandTo, ExpansionSpanMap, InFile,
    MacroCallId, MacroCallKind, MacroCallLoc, MacroDefId, MacroDefKind,
    ast::{self, AstNode},
    db::ExpandDatabase,
    mod_path::ModPath,
};

pub type EagerCallBackFn<'a> = &'a mut dyn FnMut(
    InFile<(syntax::AstPtr<ast::MacroCall>, span::FileAstId<ast::MacroCall>)>,
    MacroCallId,
);

pub fn expand_eager_macro_input(
    db: &dyn ExpandDatabase,
    krate: Crate,
    macro_call: &ast::MacroCall,
    ast_id: AstId<ast::MacroCall>,
    def: MacroDefId,
    call_site: SyntaxContext,
    resolver: &dyn Fn(&ModPath) -> Option<MacroDefId>,
    eager_callback: EagerCallBackFn<'_>,
) -> ExpandResult<Option<MacroCallId>> {
    let expand_to = ExpandTo::from_call_site(macro_call);

    // Note:
    // When `lazy_expand` is called, its *parent* file must already exist.
    // Here we store an eager macro id for the argument expanded subtree
    // for that purpose.
    let loc = MacroCallLoc {
        def,
        krate,
        kind: MacroCallKind::FnLike { ast_id, expand_to: ExpandTo::Expr, eager: None },
        ctxt: call_site,
    };
    let arg_id = db.intern_macro_call(loc);
    #[allow(deprecated)] // builtin eager macros are never derives
    let (_, _, span) = db.macro_arg(arg_id);
    let ExpandResult { value: (arg_exp, arg_exp_map), err: parse_err } =
        db.parse_macro_expansion(arg_id);

    let mut arg_map = ExpansionSpanMap::empty();

    let ExpandResult { value: expanded_eager_input, err } = {
        eager_macro_recur(
            db,
            &arg_exp_map,
            &mut arg_map,
            TextSize::new(0),
            InFile::new(arg_id.into(), arg_exp.syntax_node()),
            krate,
            call_site,
            resolver,
            eager_callback,
        )
    };
    let err = parse_err.or(err);
    if cfg!(debug_assertions) {
        arg_map.finish();
    }

    let Some((expanded_eager_input, _mapping)) = expanded_eager_input else {
        return ExpandResult { value: None, err };
    };

    let mut subtree = syntax_bridge::syntax_node_to_token_tree(
        &expanded_eager_input,
        arg_map,
        span,
        DocCommentDesugarMode::Mbe,
    );

    subtree.top_subtree_delimiter_mut().kind = crate::tt::DelimiterKind::Invisible;

    let loc = MacroCallLoc {
        def,
        krate,
        kind: MacroCallKind::FnLike {
            ast_id,
            expand_to,
            eager: Some(Arc::new(EagerCallInfo {
                arg: Arc::new(subtree),
                arg_id,
                error: err.clone(),
                span,
            })),
        },
        ctxt: call_site,
    };

    ExpandResult { value: Some(db.intern_macro_call(loc)), err }
}

fn lazy_expand(
    db: &dyn ExpandDatabase,
    def: &MacroDefId,
    macro_call: &ast::MacroCall,
    ast_id: AstId<ast::MacroCall>,
    krate: Crate,
    call_site: SyntaxContext,
    eager_callback: EagerCallBackFn<'_>,
) -> ExpandResult<(InFile<Parse<SyntaxNode>>, Arc<ExpansionSpanMap>)> {
    let expand_to = ExpandTo::from_call_site(macro_call);
    let id = def.make_call(
        db,
        krate,
        MacroCallKind::FnLike { ast_id, expand_to, eager: None },
        call_site,
    );
    eager_callback(ast_id.map(|ast_id| (AstPtr::new(macro_call), ast_id)), id);

    db.parse_macro_expansion(id).map(|parse| (InFile::new(id.into(), parse.0), parse.1))
}

fn eager_macro_recur(
    db: &dyn ExpandDatabase,
    span_map: &ExpansionSpanMap,
    expanded_map: &mut ExpansionSpanMap,
    mut offset: TextSize,
    curr: InFile<SyntaxNode>,
    krate: Crate,
    call_site: SyntaxContext,
    macro_resolver: &dyn Fn(&ModPath) -> Option<MacroDefId>,
    eager_callback: EagerCallBackFn<'_>,
) -> ExpandResult<Option<(SyntaxNode, TextSize)>> {
    let original = curr.value.clone_for_update();

    let mut replacements = Vec::new();

    // FIXME: We only report a single error inside of eager expansions
    let mut error = None;
    let mut children = original.preorder_with_tokens();

    // Collect replacement
    while let Some(child) = children.next() {
        let call = match child {
            WalkEvent::Enter(SyntaxElement::Node(child)) => match ast::MacroCall::cast(child) {
                Some(it) => {
                    children.skip_subtree();
                    it
                }
                _ => continue,
            },
            WalkEvent::Enter(_) => continue,
            WalkEvent::Leave(child) => {
                if let SyntaxElement::Token(t) = child {
                    let start = t.text_range().start();
                    offset += t.text_range().len();
                    expanded_map.push(offset, span_map.span_at(start));
                }
                continue;
            }
        };

        let def = match call.path().and_then(|path| {
            ModPath::from_src(db, path, &mut |range| span_map.span_at(range.start()).ctx)
        }) {
            Some(path) => match macro_resolver(&path) {
                Some(def) => def,
                None => {
                    let edition = krate.data(db).edition;
                    error = Some(ExpandError::other(
                        span_map.span_at(call.syntax().text_range().start()),
                        format!("unresolved macro {}", path.display(db, edition)),
                    ));
                    offset += call.syntax().text_range().len();
                    continue;
                }
            },
            None => {
                error = Some(ExpandError::other(
                    span_map.span_at(call.syntax().text_range().start()),
                    "malformed macro invocation",
                ));
                offset += call.syntax().text_range().len();
                continue;
            }
        };
        let ast_id = db.ast_id_map(curr.file_id).ast_id(&call);
        let ExpandResult { value, err } = match def.kind {
            MacroDefKind::BuiltInEager(..) => {
                let ExpandResult { value, err } = expand_eager_macro_input(
                    db,
                    krate,
                    &call,
                    curr.with_value(ast_id),
                    def,
                    call_site,
                    macro_resolver,
                    eager_callback,
                );
                match value {
                    Some(call_id) => {
                        eager_callback(
                            curr.with_value(ast_id).map(|ast_id| (AstPtr::new(&call), ast_id)),
                            call_id,
                        );
                        let ExpandResult { value: (parse, map), err: err2 } =
                            db.parse_macro_expansion(call_id);

                        map.iter().for_each(|(o, span)| expanded_map.push(o + offset, span));

                        let syntax_node = parse.syntax_node();
                        ExpandResult {
                            value: Some((
                                syntax_node.clone_for_update(),
                                offset + syntax_node.text_range().len(),
                            )),
                            err: err.or(err2),
                        }
                    }
                    None => ExpandResult { value: None, err },
                }
            }
            MacroDefKind::Declarative(_)
            | MacroDefKind::BuiltIn(..)
            | MacroDefKind::BuiltInAttr(..)
            | MacroDefKind::BuiltInDerive(..)
            | MacroDefKind::ProcMacro(..) => {
                let ExpandResult { value: (parse, tm), err } = lazy_expand(
                    db,
                    &def,
                    &call,
                    curr.with_value(ast_id),
                    krate,
                    call_site,
                    eager_callback,
                );

                // replace macro inside
                let ExpandResult { value, err: error } = eager_macro_recur(
                    db,
                    &tm,
                    expanded_map,
                    offset,
                    // FIXME: We discard parse errors here
                    parse.as_ref().map(|it| it.syntax_node()),
                    krate,
                    call_site,
                    macro_resolver,
                    eager_callback,
                );
                let err = err.or(error);

                ExpandResult { value, err }
            }
        };
        if err.is_some() {
            error = err;
        }
        // check if the whole original syntax is replaced
        if call.syntax() == &original {
            return ExpandResult { value, err: error };
        }

        match value {
            Some((insert, new_offset)) => {
                replacements.push((call, insert));
                offset = new_offset;
            }
            None => offset += call.syntax().text_range().len(),
        }
    }

    replacements.into_iter().rev().for_each(|(old, new)| ted::replace(old.syntax(), new));
    ExpandResult { value: Some((original, offset)), err: error }
}
