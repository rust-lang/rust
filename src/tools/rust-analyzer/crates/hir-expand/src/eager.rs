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
use std::sync::Arc;

use base_db::CrateId;
use syntax::{ted, SyntaxNode};

use crate::{
    ast::{self, AstNode},
    db::ExpandDatabase,
    hygiene::Hygiene,
    mod_path::ModPath,
    EagerCallInfo, ExpandError, ExpandResult, ExpandTo, InFile, MacroCallId, MacroCallKind,
    MacroCallLoc, MacroDefId, MacroDefKind, UnresolvedMacro,
};

#[derive(Debug)]
pub struct ErrorEmitted {
    _private: (),
}

pub trait ErrorSink {
    fn emit(&mut self, err: ExpandError);

    fn option<T>(
        &mut self,
        opt: Option<T>,
        error: impl FnOnce() -> ExpandError,
    ) -> Result<T, ErrorEmitted> {
        match opt {
            Some(it) => Ok(it),
            None => {
                self.emit(error());
                Err(ErrorEmitted { _private: () })
            }
        }
    }

    fn option_with<T>(
        &mut self,
        opt: impl FnOnce() -> Option<T>,
        error: impl FnOnce() -> ExpandError,
    ) -> Result<T, ErrorEmitted> {
        self.option(opt(), error)
    }

    fn result<T>(&mut self, res: Result<T, ExpandError>) -> Result<T, ErrorEmitted> {
        match res {
            Ok(it) => Ok(it),
            Err(e) => {
                self.emit(e);
                Err(ErrorEmitted { _private: () })
            }
        }
    }

    fn expand_result_option<T>(&mut self, res: ExpandResult<Option<T>>) -> Result<T, ErrorEmitted> {
        match (res.value, res.err) {
            (None, Some(err)) => {
                self.emit(err);
                Err(ErrorEmitted { _private: () })
            }
            (Some(value), opt_err) => {
                if let Some(err) = opt_err {
                    self.emit(err);
                }
                Ok(value)
            }
            (None, None) => unreachable!("`ExpandResult` without value or error"),
        }
    }
}

impl ErrorSink for &'_ mut dyn FnMut(ExpandError) {
    fn emit(&mut self, err: ExpandError) {
        self(err);
    }
}

pub fn expand_eager_macro(
    db: &dyn ExpandDatabase,
    krate: CrateId,
    macro_call: InFile<ast::MacroCall>,
    def: MacroDefId,
    resolver: &dyn Fn(ModPath) -> Option<MacroDefId>,
    diagnostic_sink: &mut dyn FnMut(ExpandError),
) -> Result<Result<MacroCallId, ErrorEmitted>, UnresolvedMacro> {
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
    // When `lazy_expand` is called, its *parent* file must be already exists.
    // Here we store an eager macro id for the argument expanded subtree here
    // for that purpose.
    let arg_id = db.intern_macro_call(MacroCallLoc {
        def,
        krate,
        eager: Some(EagerCallInfo {
            arg_or_expansion: Arc::new(parsed_args.clone()),
            included_file: None,
        }),
        kind: MacroCallKind::FnLike { ast_id: call_id, expand_to: ExpandTo::Expr },
    });

    let parsed_args = mbe::token_tree_to_syntax_node(&parsed_args, mbe::TopEntryPoint::Expr).0;
    let result = match eager_macro_recur(
        db,
        &hygiene,
        InFile::new(arg_id.as_file(), parsed_args.syntax_node()),
        krate,
        resolver,
        diagnostic_sink,
    ) {
        Ok(Ok(it)) => it,
        Ok(Err(err)) => return Ok(Err(err)),
        Err(err) => return Err(err),
    };
    let subtree = to_subtree(&result);

    if let MacroDefKind::BuiltInEager(eager, _) = def.kind {
        let res = eager.expand(db, arg_id, &subtree);
        if let Some(err) = res.err {
            diagnostic_sink(err);
        }

        let loc = MacroCallLoc {
            def,
            krate,
            eager: Some(EagerCallInfo {
                arg_or_expansion: Arc::new(res.value.subtree),
                included_file: res.value.included_file,
            }),
            kind: MacroCallKind::FnLike { ast_id: call_id, expand_to },
        };

        Ok(Ok(db.intern_macro_call(loc)))
    } else {
        panic!("called `expand_eager_macro` on non-eager macro def {def:?}");
    }
}

fn to_subtree(node: &SyntaxNode) -> crate::tt::Subtree {
    let mut subtree = mbe::syntax_node_to_token_tree(node).0;
    subtree.delimiter = crate::tt::Delimiter::unspecified();
    subtree
}

fn lazy_expand(
    db: &dyn ExpandDatabase,
    def: &MacroDefId,
    macro_call: InFile<ast::MacroCall>,
    krate: CrateId,
) -> ExpandResult<Option<InFile<SyntaxNode>>> {
    let ast_id = db.ast_id_map(macro_call.file_id).ast_id(&macro_call.value);

    let expand_to = ExpandTo::from_call_site(&macro_call.value);
    let id = def.as_lazy_macro(
        db,
        krate,
        MacroCallKind::FnLike { ast_id: macro_call.with_value(ast_id), expand_to },
    );

    let err = db.macro_expand_error(id);
    let value = db.parse_or_expand(id.as_file()).map(|node| InFile::new(id.as_file(), node));

    ExpandResult { value, err }
}

fn eager_macro_recur(
    db: &dyn ExpandDatabase,
    hygiene: &Hygiene,
    curr: InFile<SyntaxNode>,
    krate: CrateId,
    macro_resolver: &dyn Fn(ModPath) -> Option<MacroDefId>,
    mut diagnostic_sink: &mut dyn FnMut(ExpandError),
) -> Result<Result<SyntaxNode, ErrorEmitted>, UnresolvedMacro> {
    let original = curr.value.clone_for_update();

    let children = original.descendants().filter_map(ast::MacroCall::cast);
    let mut replacements = Vec::new();

    // Collect replacement
    for child in children {
        let def = match child.path().and_then(|path| ModPath::from_src(db, path, hygiene)) {
            Some(path) => macro_resolver(path.clone()).ok_or(UnresolvedMacro { path })?,
            None => {
                diagnostic_sink(ExpandError::Other("malformed macro invocation".into()));
                continue;
            }
        };
        let insert = match def.kind {
            MacroDefKind::BuiltInEager(..) => {
                let id = match expand_eager_macro(
                    db,
                    krate,
                    curr.with_value(child.clone()),
                    def,
                    macro_resolver,
                    diagnostic_sink,
                ) {
                    Ok(Ok(it)) => it,
                    Ok(Err(err)) => return Ok(Err(err)),
                    Err(err) => return Err(err),
                };
                db.parse_or_expand(id.as_file())
                    .expect("successful macro expansion should be parseable")
                    .clone_for_update()
            }
            MacroDefKind::Declarative(_)
            | MacroDefKind::BuiltIn(..)
            | MacroDefKind::BuiltInAttr(..)
            | MacroDefKind::BuiltInDerive(..)
            | MacroDefKind::ProcMacro(..) => {
                let res = lazy_expand(db, &def, curr.with_value(child.clone()), krate);
                let val = match diagnostic_sink.expand_result_option(res) {
                    Ok(it) => it,
                    Err(err) => return Ok(Err(err)),
                };

                // replace macro inside
                let hygiene = Hygiene::new(db, val.file_id);
                match eager_macro_recur(db, &hygiene, val, krate, macro_resolver, diagnostic_sink) {
                    Ok(Ok(it)) => it,
                    Ok(Err(err)) => return Ok(Err(err)),
                    Err(err) => return Err(err),
                }
            }
        };

        // check if the whole original syntax is replaced
        if child.syntax() == &original {
            return Ok(Ok(insert));
        }

        replacements.push((child, insert));
    }

    replacements.into_iter().rev().for_each(|(old, new)| ted::replace(old.syntax(), new));
    Ok(Ok(original))
}
