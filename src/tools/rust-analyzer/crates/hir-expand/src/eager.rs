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

pub fn expand_eager_macro(
    db: &dyn ExpandDatabase,
    krate: CrateId,
    macro_call: InFile<ast::MacroCall>,
    def: MacroDefId,
    resolver: &dyn Fn(ModPath) -> Option<MacroDefId>,
) -> Result<ExpandResult<Option<MacroCallId>>, UnresolvedMacro> {
    let MacroDefKind::BuiltInEager(eager, _) = def.kind else {
        panic!("called `expand_eager_macro` on non-eager macro def {def:?}")
    };
    let hygiene = Hygiene::new(db, macro_call.file_id);
    let parsed_args = macro_call
        .value
        .token_tree()
        .map(|tt| mbe::syntax_node_to_token_tree(tt.syntax()).0)
        .unwrap_or_else(tt::Subtree::empty);

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
            arg_or_expansion: Arc::new(parsed_args.clone()),
            included_file: None,
            error: None,
        })),
        kind: MacroCallKind::FnLike { ast_id: call_id, expand_to: ExpandTo::Expr },
    });

    let parsed_args = mbe::token_tree_to_syntax_node(&parsed_args, mbe::TopEntryPoint::Expr).0;
    let ExpandResult { value, mut err } = eager_macro_recur(
        db,
        &hygiene,
        InFile::new(arg_id.as_file(), parsed_args.syntax_node()),
        krate,
        resolver,
    )?;
    let Some(value ) = value else {
        return Ok(ExpandResult { value: None, err })
    };
    let subtree = {
        let mut subtree = mbe::syntax_node_to_token_tree(&value).0;
        subtree.delimiter = crate::tt::Delimiter::unspecified();
        subtree
    };

    let res = eager.expand(db, arg_id, &subtree);
    if err.is_none() {
        err = res.err;
    }

    let loc = MacroCallLoc {
        def,
        krate,
        eager: Some(Box::new(EagerCallInfo {
            arg_or_expansion: Arc::new(res.value.subtree),
            included_file: res.value.included_file,
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

    let file_id = id.as_file();
    db.parse_or_expand_with_err(file_id).map(|parse| InFile::new(file_id, parse))
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
                error = Some(ExpandError::Other("malformed macro invocation".into()));
                continue;
            }
        };
        let ExpandResult { value, err } = match def.kind {
            MacroDefKind::BuiltInEager(..) => {
                let id = match expand_eager_macro(
                    db,
                    krate,
                    curr.with_value(child.clone()),
                    def,
                    macro_resolver,
                ) {
                    Ok(it) => it,
                    Err(err) => return Err(err),
                };
                id.map(|call| {
                    call.map(|call| db.parse_or_expand(call.as_file()).clone_for_update())
                })
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
                let err = if err.is_none() { error } else { err };
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
