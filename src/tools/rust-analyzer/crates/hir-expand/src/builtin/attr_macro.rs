//! Builtin attributes.
use intern::sym;
use span::{MacroCallId, Span};

use crate::{db::ExpandDatabase, name, tt, ExpandResult, MacroCallKind};

use super::quote;

macro_rules! register_builtin {
    ($(($name:ident, $variant:ident) => $expand:ident),* ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinAttrExpander {
            $($variant),*
        }

        impl BuiltinAttrExpander {
            pub fn expander(&self) -> fn (&dyn ExpandDatabase, MacroCallId, &tt::Subtree, Span) -> ExpandResult<tt::Subtree>  {
                match *self {
                    $( BuiltinAttrExpander::$variant => $expand, )*
                }
            }

            fn find_by_name(name: &name::Name) -> Option<Self> {
                match name {
                    $( id if id == &sym::$name => Some(BuiltinAttrExpander::$variant), )*
                     _ => None,
                }
            }
        }

    };
}

impl BuiltinAttrExpander {
    pub fn expand(
        &self,
        db: &dyn ExpandDatabase,
        id: MacroCallId,
        tt: &tt::Subtree,
        span: Span,
    ) -> ExpandResult<tt::Subtree> {
        self.expander()(db, id, tt, span)
    }

    pub fn is_derive(self) -> bool {
        matches!(self, BuiltinAttrExpander::Derive | BuiltinAttrExpander::DeriveConst)
    }
    pub fn is_test(self) -> bool {
        matches!(self, BuiltinAttrExpander::Test)
    }
    pub fn is_bench(self) -> bool {
        matches!(self, BuiltinAttrExpander::Bench)
    }
}

register_builtin! {
    (bench, Bench) => dummy_gate_test_expand,
    (cfg_accessible, CfgAccessible) => dummy_attr_expand,
    (cfg_eval, CfgEval) => dummy_attr_expand,
    (derive, Derive) => derive_expand,
    // derive const is equivalent to derive for our proposes.
    (derive_const, DeriveConst) => derive_expand,
    (global_allocator, GlobalAllocator) => dummy_attr_expand,
    (test, Test) => dummy_gate_test_expand,
    (test_case, TestCase) => dummy_gate_test_expand
}

pub fn find_builtin_attr(ident: &name::Name) -> Option<BuiltinAttrExpander> {
    BuiltinAttrExpander::find_by_name(ident)
}

fn dummy_attr_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
    _span: Span,
) -> ExpandResult<tt::Subtree> {
    ExpandResult::ok(tt.clone())
}

fn dummy_gate_test_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let result = quote::quote! { span=>
        #[cfg(test)]
        #tt
    };
    ExpandResult::ok(result)
}

/// We generate a very specific expansion here, as we do not actually expand the `#[derive]` attribute
/// itself in name res, but we do want to expand it to something for the IDE layer, so that the input
/// derive attributes can be downmapped, and resolved as proper paths.
/// This is basically a hack, that simplifies the hacks we need in a lot of ide layer places to
/// somewhat inconsistently resolve derive attributes.
///
/// As such, we expand `#[derive(Foo, bar::Bar)]` into
/// ```
///  #![Foo]
///  #![bar::Bar]
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
fn derive_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let loc = db.lookup_intern_macro_call(id);
    let derives = match &loc.kind {
        MacroCallKind::Attr { attr_args: Some(attr_args), .. } if loc.def.is_attribute_derive() => {
            attr_args
        }
        _ => {
            return ExpandResult::ok(tt::Subtree::empty(tt::DelimSpan { open: span, close: span }))
        }
    };
    pseudo_derive_attr_expansion(tt, derives, span)
}

pub fn pseudo_derive_attr_expansion(
    _: &tt::Subtree,
    args: &tt::Subtree,
    call_site: Span,
) -> ExpandResult<tt::Subtree> {
    let mk_leaf = |char| {
        tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct {
            char,
            spacing: tt::Spacing::Alone,
            span: call_site,
        }))
    };

    let mut token_trees = Vec::new();
    for tt in args
        .token_trees
        .split(|tt| matches!(tt, tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: ',', .. }))))
    {
        token_trees.push(mk_leaf('#'));
        token_trees.push(mk_leaf('!'));
        token_trees.push(mk_leaf('['));
        token_trees.extend(tt.iter().cloned());
        token_trees.push(mk_leaf(']'));
    }
    ExpandResult::ok(tt::Subtree {
        delimiter: args.delimiter,
        token_trees: token_trees.into_boxed_slice(),
    })
}
