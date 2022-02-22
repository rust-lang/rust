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

/// We generate a very specific expansion here, as we do not actually expand the `#[derive]` attribute
/// itself in name res, but we do want to expand it to something for the IDE layer, so that the input
/// derive attributes can be downmapped, and resolved as proper paths.
/// This is basically a hack, that simplifies the hacks we need in a lot of ide layer places to
/// somewhat inconsistently resolve derive attributes.
///
/// As such, we expand `#[derive(Foo, bar::Bar)]` into
/// ```
///  #[Foo]
///  #[bar::Bar]
///  ();
/// ```
/// which allows fallback path resolution in hir::Semantics to properly identify our derives.
/// Since we do not expand the attribute in nameres though, we keep the original item.
///
/// The ideal expansion here would be for the `#[derive]` to re-emit the annotated item and somehow
/// use the input paths in its output as well.
/// But that would bring two problems with it, for one every derive would duplicate the item token tree
/// wasting a lot of memory, and it would also require some way to use a path in a way that makes it
/// always resolve as a derive without nameres recollecting them.
/// So this hacky approach is a lot more friendly for us, though it does require a bit of support in
/// [`hir::Semantics`] to make this work.
fn derive_attr_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let loc = db.lookup_intern_macro_call(id);
    let derives = match &loc.kind {
        MacroCallKind::Attr { attr_args, .. } => &attr_args.0,
        _ => return ExpandResult::ok(tt.clone()),
    };

    let mk_leaf = |char| {
        tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct {
            char,
            spacing: tt::Spacing::Alone,
            id: tt::TokenId::unspecified(),
        }))
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
        token_trees.push(mk_leaf('#'));
        token_trees.push(mk_leaf('['));
        token_trees.extend(group.cloned().map(tt::TokenTree::Leaf));
        token_trees.push(mk_leaf(']'));
    }
    token_trees.push(mk_leaf('('));
    token_trees.push(mk_leaf(')'));
    token_trees.push(mk_leaf(';'));
    ExpandResult::ok(tt::Subtree { delimiter: tt.delimiter, token_trees })
}
