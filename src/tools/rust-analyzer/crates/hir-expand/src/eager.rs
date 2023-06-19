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
use base_db::CrateId;
use syntax::{ted, Parse, SyntaxNode};
use triomphe::Arc;

use crate::{
    ast::{self, AstNode},
    db::ExpandDatabase,
    hygiene::Hygiene,
    mod_path::ModPath,
    EagerCallInfo, ExpandError, ExpandResult, ExpandTo, InFile, MacroCallId, MacroCallKind,
    MacroCallLoc, MacroDefId, MacroDefKind, UnresolvedMacro,
};

pub fn expand_eager_macro_input(
    db: &dyn ExpandDatabase,
    krate: CrateId,
    macro_call: InFile<ast::MacroCall>,
    def: MacroDefId,
    resolver: &dyn Fn(ModPath) -> Option<MacroDefId>,
) -> Result<ExpandResult<Option<MacroCallId>>, UnresolvedMacro> {
    assert!(matches!(def.kind, MacroDefKind::BuiltInEager(..)));
    let token_tree = macro_call.value.token_tree();

    let Some(token_tree) = token_tree else {
        return Ok(ExpandResult { value: None, err:
            Some(ExpandError::other(
                "invalid token tree"
            )),
        });
    };
    let (parsed_args, arg_token_map) = mbe::syntax_node_to_token_tree(token_tree.syntax());

    let ast_map = db.ast_id_map(macro_call.file_id);
    let call_id = InFile::new(macro_call.file_id, ast_map.ast_id(&macro_call.value));
    let expand_to = ExpandTo::from_call_site(&macro_call.value);

    // Note:
    // When `lazy_expand` is called, its *parent* file must already exist.
    // Here we store an eager macro id for the argument expanded subtree
    // for that purpose.
    let arg_id = db.intern_macro_call(MacroCallLoc {
        def,
        krate,
        eager: Some(Box::new(EagerCallInfo {
            arg: Arc::new((parsed_args, arg_token_map)),
            arg_id: None,
            error: None,
        })),
        kind: MacroCallKind::FnLike { ast_id: call_id, expand_to: ExpandTo::Expr },
    });
    let arg_as_expr = match db.macro_arg_text(arg_id) {
        Some(it) => it,
        None => {
            return Ok(ExpandResult {
                value: None,
                err: Some(ExpandError::other("invalid token tree")),
            })
        }
    };
    let ExpandResult { value: expanded_eager_input, err } = eager_macro_recur(
        db,
        &Hygiene::new(db, macro_call.file_id),
        InFile::new(arg_id.as_file(), SyntaxNode::new_root(arg_as_expr)),
        krate,
        resolver,
    )?;
    let Some(expanded_eager_input) = expanded_eager_input else {
        return Ok(ExpandResult { value: None, err })
    };
    let (mut subtree, token_map) = mbe::syntax_node_to_token_tree(&expanded_eager_input);
    subtree.delimiter = crate::tt::Delimiter::unspecified();

    let loc = MacroCallLoc {
        def,
        krate,
        eager: Some(Box::new(EagerCallInfo {
            arg: Arc::new((subtree, token_map)),
            arg_id: Some(arg_id),
            error: err.clone(),
        })),
        kind: MacroCallKind::FnLike { ast_id: call_id, expand_to },
    };

    Ok(ExpandResult { value: Some(db.intern_macro_call(loc)), err })
}

fn lazy_expand(
    db: &dyn ExpandDatabase,
    def: &MacroDefId,
    macro_call: InFile<ast::MacroCall>,
    krate: CrateId,
) -> ExpandResult<InFile<Parse<SyntaxNode>>> {
    let ast_id = db.ast_id_map(macro_call.file_id).ast_id(&macro_call.value);

    let expand_to = ExpandTo::from_call_site(&macro_call.value);
    let id = def.as_lazy_macro(
        db,
        krate,
        MacroCallKind::FnLike { ast_id: macro_call.with_value(ast_id), expand_to },
    );

    let macro_file = id.as_macro_file();

    db.parse_macro_expansion(macro_file).map(|parse| InFile::new(macro_file.into(), parse.0))
}

fn eager_macro_recur(
    db: &dyn ExpandDatabase,
    hygiene: &Hygiene,
    curr: InFile<SyntaxNode>,
    krate: CrateId,
    macro_resolver: &dyn Fn(ModPath) -> Option<MacroDefId>,
) -> Result<ExpandResult<Option<SyntaxNode>>, UnresolvedMacro> {
    let original = curr.value.clone_for_update();

    let children = original.descendants().filter_map(ast::MacroCall::cast);
    let mut replacements = Vec::new();

    // Note: We only report a single error inside of eager expansions
    let mut error = None;

    // Collect replacement
    for child in children {
        let def = match child.path().and_then(|path| ModPath::from_src(db, path, hygiene)) {
            Some(path) => macro_resolver(path.clone()).ok_or(UnresolvedMacro { path })?,
            None => {
                error = Some(ExpandError::other("malformed macro invocation"));
                continue;
            }
        };
        let ExpandResult { value, err } = match def.kind {
            MacroDefKind::BuiltInEager(..) => {
                let ExpandResult { value, err } = match expand_eager_macro_input(
                    db,
                    krate,
                    curr.with_value(child.clone()),
                    def,
                    macro_resolver,
                ) {
                    Ok(it) => it,
                    Err(err) => return Err(err),
                };
                match value {
                    Some(call) => {
                        let ExpandResult { value, err: err2 } =
                            db.parse_macro_expansion(call.as_macro_file());
                        ExpandResult {
                            value: Some(value.0.syntax_node().clone_for_update()),
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
                let ExpandResult { value, err } =
                    lazy_expand(db, &def, curr.with_value(child.clone()), krate);

                // replace macro inside
                let hygiene = Hygiene::new(db, value.file_id);
                let ExpandResult { value, err: error } = eager_macro_recur(
                    db,
                    &hygiene,
                    // FIXME: We discard parse errors here
                    value.map(|it| it.syntax_node()),
                    krate,
                    macro_resolver,
                )?;
                let err = err.or(error);
                ExpandResult { value, err }
            }
        };
        if err.is_some() {
            error = err;
        }
        // check if the whole original syntax is replaced
        if child.syntax() == &original {
            return Ok(ExpandResult { value, err: error });
        }

        if let Some(insert) = value {
            replacements.push((child, insert));
        }
    }

    replacements.into_iter().rev().for_each(|(old, new)| ted::replace(old.syntax(), new));
    Ok(ExpandResult { value: Some(original), err: error })
}
