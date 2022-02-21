//! Builtin attributes.

use itertools::Itertools;
use syntax::ast;

use crate::{
    db::AstDatabase, name, AstId, CrateId, ExpandResult, MacroCallId, MacroCallKind, MacroDefId,
    MacroDefKind,
};

macro_rules! register_builtin {
    ( $(($name:ident, $variant:ident) => $expand:ident),* ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinAttrExpander {
            $($variant),*
        }

        impl BuiltinAttrExpander {
            pub fn expand(
                &self,
                db: &dyn AstDatabase,
                id: MacroCallId,
                tt: &tt::Subtree,
            ) -> ExpandResult<tt::Subtree> {
                let expander = match *self {
                    $( BuiltinAttrExpander::$variant => $expand, )*
                };
                expander(db, id, tt)
            }

            fn find_by_name(name: &name::Name) -> Option<Self> {
                match name {
                    $( id if id == &name::name![$name] => Some(BuiltinAttrExpander::$variant), )*
                     _ => None,
                }
            }
        }

    };
}

impl BuiltinAttrExpander {
    pub fn is_derive(self) -> bool {
        matches!(self, BuiltinAttrExpander::Derive)
    }
    pub fn is_test(self) -> bool {
        matches!(self, BuiltinAttrExpander::Test)
    }
    pub fn is_bench(self) -> bool {
        matches!(self, BuiltinAttrExpander::Bench)
    }
}

register_builtin! {
    (bench, Bench) => dummy_attr_expand,
    (cfg_accessible, CfgAccessible) => dummy_attr_expand,
    (cfg_eval, CfgEval) => dummy_attr_expand,
    (derive, Derive) => derive_attr_expand,
    (global_allocator, GlobalAllocator) => dummy_attr_expand,
    (test, Test) => dummy_attr_expand,
    (test_case, TestCase) => dummy_attr_expand
}

pub fn find_builtin_attr(
    ident: &name::Name,
    krate: CrateId,
    ast_id: AstId<ast::Macro>,
) -> Option<MacroDefId> {
    let expander = BuiltinAttrExpander::find_by_name(ident)?;
    Some(MacroDefId {
        krate,
        kind: MacroDefKind::BuiltInAttr(expander, ast_id),
        local_inner: false,
    })
}

fn dummy_attr_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    ExpandResult::ok(tt.clone())
}

fn derive_attr_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    // we generate a very specific expansion here, as we do not actually expand the `#[derive]` attribute
    // itself in name res, but we do want to expand it to something for the IDE layer, so that the input
    // derive attributes can be downmapped, and resolved
    // This is basically a hack, to get rid of hacks in the IDE layer that slowly accumulate more and more
    // in various places.

    // we transform the token tree of `#[derive(Foo, bar::Bar)]` into
    // ```
    //  #[Foo]
    //  #[bar::Bar]
    //  ();
    // ```
    // which allows fallback path resolution in hir::Semantics to properly identify our derives
    let loc = db.lookup_intern_macro_call(id);
    let derives = match &loc.kind {
        MacroCallKind::Attr { attr_args, .. } => &attr_args.0,
        _ => return ExpandResult::ok(tt.clone()),
    };

    let mut token_trees = Vec::new();
    for (comma, group) in &derives
        .token_trees
        .iter()
        .filter_map(|tt| match tt {
            tt::TokenTree::Leaf(l) => Some(l),
            tt::TokenTree::Subtree(_) => None,
        })
        .group_by(|l| matches!(l, tt::Leaf::Punct(tt::Punct { char: ',', .. })))
    {
        if comma {
            continue;
        }
        let wrap = |leaf| tt::TokenTree::Leaf(tt::Leaf::Punct(leaf));
        token_trees.push(wrap(tt::Punct {
            char: '#',
            spacing: tt::Spacing::Alone,
            id: tt::TokenId::unspecified(),
        }));
        token_trees.push(wrap(tt::Punct {
            char: '[',
            spacing: tt::Spacing::Alone,
            id: tt::TokenId::unspecified(),
        }));
        token_trees.extend(group.cloned().map(tt::TokenTree::Leaf));
        token_trees.push(wrap(tt::Punct {
            char: ']',
            spacing: tt::Spacing::Alone,
            id: tt::TokenId::unspecified(),
        }));
        token_trees.push(wrap(tt::Punct {
            char: '(',
            spacing: tt::Spacing::Alone,
            id: tt::TokenId::unspecified(),
        }));
        token_trees.push(wrap(tt::Punct {
            char: ')',
            spacing: tt::Spacing::Alone,
            id: tt::TokenId::unspecified(),
        }));
        token_trees.push(wrap(tt::Punct {
            char: ';',
            spacing: tt::Spacing::Alone,
            id: tt::TokenId::unspecified(),
        }));
    }
    ExpandResult::ok(tt::Subtree { delimiter: tt.delimiter, token_trees })
}
