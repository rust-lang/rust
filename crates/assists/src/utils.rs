//! Assorted functions shared by several assists.
pub(crate) mod insert_use;
pub(crate) mod import_assets;

use std::ops;

use hir::{Crate, Enum, HasSource, Module, ScopeDef, Semantics, Trait};
use ide_db::RootDatabase;
use itertools::Itertools;
use syntax::{
    ast::edit::AstNodeEdit,
    ast::AttrsOwner,
    ast::NameOwner,
    ast::{self, edit, make, ArgListOwner},
    AstNode, Direction,
    SyntaxKind::*,
    SyntaxNode, TextSize, T,
};

use crate::{
    assist_config::SnippetCap,
    ast_transform::{self, AstTransform, QualifyPaths, SubstituteTypeParams},
};

pub use insert_use::{insert_use, ImportScope, MergeBehaviour};

pub fn mod_path_to_ast(path: &hir::ModPath) -> ast::Path {
    let _p = profile::span("mod_path_to_ast");
    let mut segments = Vec::new();
    let mut is_abs = false;
    match path.kind {
        hir::PathKind::Plain => {}
        hir::PathKind::Super(0) => segments.push(make::path_segment_self()),
        hir::PathKind::Super(n) => segments.extend((0..n).map(|_| make::path_segment_super())),
        hir::PathKind::DollarCrate(_) | hir::PathKind::Crate => {
            segments.push(make::path_segment_crate())
        }
        hir::PathKind::Abs => is_abs = true,
    }

    segments.extend(
        path.segments
            .iter()
            .map(|segment| make::path_segment(make::name_ref(&segment.to_string()))),
    );
    make::path_from_segments(segments, is_abs)
}

pub(crate) fn unwrap_trivial_block(block: ast::BlockExpr) -> ast::Expr {
    extract_trivial_expression(&block)
        .filter(|expr| !expr.syntax().text().contains_char('\n'))
        .unwrap_or_else(|| block.into())
}

pub fn extract_trivial_expression(block: &ast::BlockExpr) -> Option<ast::Expr> {
    let has_anything_else = |thing: &SyntaxNode| -> bool {
        let mut non_trivial_children =
            block.syntax().children_with_tokens().filter(|it| match it.kind() {
                WHITESPACE | T!['{'] | T!['}'] => false,
                _ => it.as_node() != Some(thing),
            });
        non_trivial_children.next().is_some()
    };

    if let Some(expr) = block.expr() {
        if has_anything_else(expr.syntax()) {
            return None;
        }
        return Some(expr);
    }
    // Unwrap `{ continue; }`
    let (stmt,) = block.statements().next_tuple()?;
    if let ast::Stmt::ExprStmt(expr_stmt) = stmt {
        if has_anything_else(expr_stmt.syntax()) {
            return None;
        }
        let expr = expr_stmt.expr()?;
        match expr.syntax().kind() {
            CONTINUE_EXPR | BREAK_EXPR | RETURN_EXPR => return Some(expr),
            _ => (),
        }
    }
    None
}

/// This is a method with a heuristics to support test methods annotated with custom test annotations, such as
/// `#[test_case(...)]`, `#[tokio::test]` and similar.
/// Also a regular `#[test]` annotation is supported.
///
/// It may produce false positives, for example, `#[wasm_bindgen_test]` requires a different command to run the test,
/// but it's better than not to have the runnables for the tests at all.
pub fn test_related_attribute(fn_def: &ast::Fn) -> Option<ast::Attr> {
    fn_def.attrs().find_map(|attr| {
        let path = attr.path()?;
        if path.syntax().text().to_string().contains("test") {
            Some(attr)
        } else {
            None
        }
    })
}

#[derive(Copy, Clone, PartialEq)]
pub enum DefaultMethods {
    Only,
    No,
}

pub fn filter_assoc_items(
    db: &RootDatabase,
    items: &[hir::AssocItem],
    default_methods: DefaultMethods,
) -> Vec<ast::AssocItem> {
    fn has_def_name(item: &ast::AssocItem) -> bool {
        match item {
            ast::AssocItem::Fn(def) => def.name(),
            ast::AssocItem::TypeAlias(def) => def.name(),
            ast::AssocItem::Const(def) => def.name(),
            ast::AssocItem::MacroCall(_) => None,
        }
        .is_some()
    };

    items
        .iter()
        .map(|i| match i {
            hir::AssocItem::Function(i) => ast::AssocItem::Fn(i.source(db).value),
            hir::AssocItem::TypeAlias(i) => ast::AssocItem::TypeAlias(i.source(db).value),
            hir::AssocItem::Const(i) => ast::AssocItem::Const(i.source(db).value),
        })
        .filter(has_def_name)
        .filter(|it| match it {
            ast::AssocItem::Fn(def) => matches!(
                (default_methods, def.body()),
                (DefaultMethods::Only, Some(_)) | (DefaultMethods::No, None)
            ),
            _ => default_methods == DefaultMethods::No,
        })
        .collect::<Vec<_>>()
}

pub fn add_trait_assoc_items_to_impl(
    sema: &hir::Semantics<ide_db::RootDatabase>,
    items: Vec<ast::AssocItem>,
    trait_: hir::Trait,
    impl_def: ast::Impl,
    target_scope: hir::SemanticsScope,
) -> (ast::Impl, ast::AssocItem) {
    let impl_item_list = impl_def.assoc_item_list().unwrap_or_else(make::assoc_item_list);

    let n_existing_items = impl_item_list.assoc_items().count();
    let source_scope = sema.scope_for_def(trait_);
    let ast_transform = QualifyPaths::new(&target_scope, &source_scope)
        .or(SubstituteTypeParams::for_trait_impl(&source_scope, trait_, impl_def.clone()));

    let items = items
        .into_iter()
        .map(|it| ast_transform::apply(&*ast_transform, it))
        .map(|it| match it {
            ast::AssocItem::Fn(def) => ast::AssocItem::Fn(add_body(def)),
            ast::AssocItem::TypeAlias(def) => ast::AssocItem::TypeAlias(def.remove_bounds()),
            _ => it,
        })
        .map(|it| edit::remove_attrs_and_docs(&it));

    let new_impl_item_list = impl_item_list.append_items(items);
    let new_impl_def = impl_def.with_assoc_item_list(new_impl_item_list);
    let first_new_item =
        new_impl_def.assoc_item_list().unwrap().assoc_items().nth(n_existing_items).unwrap();
    return (new_impl_def, first_new_item);

    fn add_body(fn_def: ast::Fn) -> ast::Fn {
        match fn_def.body() {
            Some(_) => fn_def,
            None => {
                let body =
                    make::block_expr(None, Some(make::expr_todo())).indent(edit::IndentLevel(1));
                fn_def.with_body(body)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Cursor<'a> {
    Replace(&'a SyntaxNode),
    Before(&'a SyntaxNode),
}

impl<'a> Cursor<'a> {
    fn node(self) -> &'a SyntaxNode {
        match self {
            Cursor::Replace(node) | Cursor::Before(node) => node,
        }
    }
}

pub(crate) fn render_snippet(_cap: SnippetCap, node: &SyntaxNode, cursor: Cursor) -> String {
    assert!(cursor.node().ancestors().any(|it| it == *node));
    let range = cursor.node().text_range() - node.text_range().start();
    let range: ops::Range<usize> = range.into();

    let mut placeholder = cursor.node().to_string();
    escape(&mut placeholder);
    let tab_stop = match cursor {
        Cursor::Replace(placeholder) => format!("${{0:{}}}", placeholder),
        Cursor::Before(placeholder) => format!("$0{}", placeholder),
    };

    let mut buf = node.to_string();
    buf.replace_range(range, &tab_stop);
    return buf;

    fn escape(buf: &mut String) {
        stdx::replace(buf, '{', r"\{");
        stdx::replace(buf, '}', r"\}");
        stdx::replace(buf, '$', r"\$");
    }
}

pub(crate) fn vis_offset(node: &SyntaxNode) -> TextSize {
    node.children_with_tokens()
        .find(|it| !matches!(it.kind(), WHITESPACE | COMMENT | ATTR))
        .map(|it| it.text_range().start())
        .unwrap_or_else(|| node.text_range().start())
}

pub(crate) fn invert_boolean_expression(expr: ast::Expr) -> ast::Expr {
    if let Some(expr) = invert_special_case(&expr) {
        return expr;
    }
    make::expr_prefix(T![!], expr)
}

fn invert_special_case(expr: &ast::Expr) -> Option<ast::Expr> {
    match expr {
        ast::Expr::BinExpr(bin) => match bin.op_kind()? {
            ast::BinOp::NegatedEqualityTest => bin.replace_op(T![==]).map(|it| it.into()),
            ast::BinOp::EqualityTest => bin.replace_op(T![!=]).map(|it| it.into()),
            _ => None,
        },
        ast::Expr::MethodCallExpr(mce) => {
            let receiver = mce.receiver()?;
            let method = mce.name_ref()?;
            let arg_list = mce.arg_list()?;

            let method = match method.text().as_str() {
                "is_some" => "is_none",
                "is_none" => "is_some",
                "is_ok" => "is_err",
                "is_err" => "is_ok",
                _ => return None,
            };
            Some(make::expr_method_call(receiver, method, arg_list))
        }
        ast::Expr::PrefixExpr(pe) if pe.op_kind()? == ast::PrefixOp::Not => pe.expr(),
        // FIXME:
        // ast::Expr::Literal(true | false )
        _ => None,
    }
}

/// Helps with finding well-know things inside the standard library. This is
/// somewhat similar to the known paths infra inside hir, but it different; We
/// want to make sure that IDE specific paths don't become interesting inside
/// the compiler itself as well.
pub struct FamousDefs<'a, 'b>(pub &'a Semantics<'b, RootDatabase>, pub Option<Crate>);

#[allow(non_snake_case)]
impl FamousDefs<'_, '_> {
    pub const FIXTURE: &'static str = r#"//- /libcore.rs crate:core
pub mod convert {
    pub trait From<T> {
        fn from(t: T) -> Self;
    }
}

pub mod default {
    pub trait Default {
       fn default() -> Self;
    }
}

pub mod iter {
    pub use self::traits::{collect::IntoIterator, iterator::Iterator};
    mod traits {
        pub(crate) mod iterator {
            use crate::option::Option;
            pub trait Iterator {
                type Item;
                fn next(&mut self) -> Option<Self::Item>;
                fn by_ref(&mut self) -> &mut Self {
                    self
                }
                fn take(self, n: usize) -> crate::iter::Take<Self> {
                    crate::iter::Take { inner: self }
                }
            }

            impl<I: Iterator> Iterator for &mut I {
                type Item = I::Item;
                fn next(&mut self) -> Option<I::Item> {
                    (**self).next()
                }
            }
        }
        pub(crate) mod collect {
            pub trait IntoIterator {
                type Item;
            }
        }
    }

    pub use self::sources::*;
    pub(crate) mod sources {
        use super::Iterator;
        use crate::option::Option::{self, *};
        pub struct Repeat<A> {
            element: A,
        }

        pub fn repeat<T>(elt: T) -> Repeat<T> {
            Repeat { element: elt }
        }

        impl<A> Iterator for Repeat<A> {
            type Item = A;

            fn next(&mut self) -> Option<A> {
                None
            }
        }
    }

    pub use self::adapters::*;
    pub(crate) mod adapters {
        use super::Iterator;
        use crate::option::Option::{self, *};
        pub struct Take<I> { pub(crate) inner: I }
        impl<I> Iterator for Take<I> where I: Iterator {
            type Item = <I as Iterator>::Item;
            fn next(&mut self) -> Option<<I as Iterator>::Item> {
                None
            }
        }
    }
}

pub mod option {
    pub enum Option<T> { None, Some(T)}
}

pub mod prelude {
    pub use crate::{convert::From, iter::{IntoIterator, Iterator}, option::Option::{self, *}, default::Default};
}
#[prelude_import]
pub use prelude::*;
"#;

    pub fn core(&self) -> Option<Crate> {
        self.find_crate("core")
    }

    pub(crate) fn core_convert_From(&self) -> Option<Trait> {
        self.find_trait("core:convert:From")
    }

    pub(crate) fn core_option_Option(&self) -> Option<Enum> {
        self.find_enum("core:option:Option")
    }

    pub fn core_default_Default(&self) -> Option<Trait> {
        self.find_trait("core:default:Default")
    }

    pub fn core_iter_Iterator(&self) -> Option<Trait> {
        self.find_trait("core:iter:traits:iterator:Iterator")
    }

    pub fn core_iter(&self) -> Option<Module> {
        self.find_module("core:iter")
    }

    fn find_trait(&self, path: &str) -> Option<Trait> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Trait(it)) => Some(it),
            _ => None,
        }
    }

    fn find_enum(&self, path: &str) -> Option<Enum> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Adt(hir::Adt::Enum(it))) => Some(it),
            _ => None,
        }
    }

    fn find_module(&self, path: &str) -> Option<Module> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Module(it)) => Some(it),
            _ => None,
        }
    }

    fn find_crate(&self, name: &str) -> Option<Crate> {
        let krate = self.1?;
        let db = self.0.db;
        let res =
            krate.dependencies(db).into_iter().find(|dep| dep.name.to_string() == name)?.krate;
        Some(res)
    }

    fn find_def(&self, path: &str) -> Option<ScopeDef> {
        let db = self.0.db;
        let mut path = path.split(':');
        let trait_ = path.next_back()?;
        let std_crate = path.next()?;
        let std_crate = self.find_crate(std_crate)?;
        let mut module = std_crate.root_module(db);
        for segment in path {
            module = module.children(db).find_map(|child| {
                let name = child.name(db)?;
                if name.to_string() == segment {
                    Some(child)
                } else {
                    None
                }
            })?;
        }
        let def =
            module.scope(db, None).into_iter().find(|(name, _def)| name.to_string() == trait_)?.1;
        Some(def)
    }
}

pub(crate) fn next_prev() -> impl Iterator<Item = Direction> {
    [Direction::Next, Direction::Prev].iter().copied()
}
