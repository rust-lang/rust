use crate::hir::map::definitions::*;
use crate::hir::def_id::{CRATE_DEF_INDEX, DefIndex};
use crate::session::CrateDisambiguator;

use syntax::ast::*;
use syntax::ext::hygiene::Mark;
use syntax::visit;
use syntax::symbol::{kw, sym};
use syntax::parse::token::{self, Token};
use syntax_pos::Span;

/// Creates `DefId`s for nodes in the AST.
pub struct DefCollector<'a> {
    definitions: &'a mut Definitions,
    parent_def: Option<DefIndex>,
    expansion: Mark,
    pub visit_macro_invoc: Option<&'a mut dyn FnMut(MacroInvocationData)>,
}

pub struct MacroInvocationData {
    pub mark: Mark,
    pub def_index: DefIndex,
}

impl<'a> DefCollector<'a> {
    pub fn new(definitions: &'a mut Definitions, expansion: Mark) -> Self {
        DefCollector {
            definitions,
            expansion,
            parent_def: None,
            visit_macro_invoc: None,
        }
    }

    pub fn collect_root(&mut self,
                        crate_name: &str,
                        crate_disambiguator: CrateDisambiguator) {
        let root = self.definitions.create_root_def(crate_name,
                                                    crate_disambiguator);
        assert_eq!(root, CRATE_DEF_INDEX);
        self.parent_def = Some(root);
    }

    fn create_def(&mut self,
                  node_id: NodeId,
                  data: DefPathData,
                  span: Span)
                  -> DefIndex {
        let parent_def = self.parent_def.unwrap();
        debug!("create_def(node_id={:?}, data={:?}, parent_def={:?})", node_id, data, parent_def);
        self.definitions
            .create_def_with_parent(parent_def, node_id, data, self.expansion, span)
    }

    pub fn with_parent<F: FnOnce(&mut Self)>(&mut self, parent_def: DefIndex, f: F) {
        let parent = self.parent_def;
        self.parent_def = Some(parent_def);
        f(self);
        self.parent_def = parent;
    }

    fn visit_async_fn(
        &mut self,
        id: NodeId,
        name: Name,
        span: Span,
        header: &FnHeader,
        generics: &'a Generics,
        decl: &'a FnDecl,
        body: &'a Block,
    ) {
        let (closure_id, return_impl_trait_id) = match header.asyncness.node {
            IsAsync::Async {
                closure_id,
                return_impl_trait_id,
            } => (closure_id, return_impl_trait_id),
            _ => unreachable!(),
        };

        // For async functions, we need to create their inner defs inside of a
        // closure to match their desugared representation.
        let fn_def_data = DefPathData::ValueNs(name.as_interned_str());
        let fn_def = self.create_def(id, fn_def_data, span);
        return self.with_parent(fn_def, |this| {
            this.create_def(return_impl_trait_id, DefPathData::ImplTrait, span);

            visit::walk_generics(this, generics);
            visit::walk_fn_decl(this, decl);

            let closure_def = this.create_def(
                closure_id, DefPathData::ClosureExpr, span,
            );
            this.with_parent(closure_def, |this| {
                visit::walk_block(this, body);
            })
        })
    }

    fn visit_macro_invoc(&mut self, id: NodeId) {
        if let Some(ref mut visit) = self.visit_macro_invoc {
            visit(MacroInvocationData {
                mark: id.placeholder_to_mark(),
                def_index: self.parent_def.unwrap(),
            })
        }
    }
}

impl<'a> visit::Visitor<'a> for DefCollector<'a> {
    fn visit_item(&mut self, i: &'a Item) {
        debug!("visit_item: {:?}", i);

        // Pick the def data. This need not be unique, but the more
        // information we encapsulate into, the better
        let def_data = match i.node {
            ItemKind::Impl(..) => DefPathData::Impl,
            ItemKind::Mod(..) if i.ident.name == kw::Invalid => {
                return visit::walk_item(self, i);
            }
            ItemKind::Mod(..) | ItemKind::Trait(..) | ItemKind::TraitAlias(..) |
            ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Union(..) |
            ItemKind::Existential(..) | ItemKind::ExternCrate(..) | ItemKind::ForeignMod(..) |
            ItemKind::Ty(..) => DefPathData::TypeNs(i.ident.as_interned_str()),
            ItemKind::Fn(
                ref decl,
                ref header,
                ref generics,
                ref body,
            ) if header.asyncness.node.is_async() => {
                return self.visit_async_fn(
                    i.id,
                    i.ident.name,
                    i.span,
                    header,
                    generics,
                    decl,
                    body,
                )
            }
            ItemKind::Static(..) | ItemKind::Const(..) | ItemKind::Fn(..) =>
                DefPathData::ValueNs(i.ident.as_interned_str()),
            ItemKind::MacroDef(..) => DefPathData::MacroNs(i.ident.as_interned_str()),
            ItemKind::Mac(..) => return self.visit_macro_invoc(i.id),
            ItemKind::GlobalAsm(..) => DefPathData::Misc,
            ItemKind::Use(..) => {
                return visit::walk_item(self, i);
            }
        };
        let def = self.create_def(i.id, def_data, i.span);

        self.with_parent(def, |this| {
            match i.node {
                ItemKind::Struct(ref struct_def, _) | ItemKind::Union(ref struct_def, _) => {
                    // If this is a unit or tuple-like struct, register the constructor.
                    if let Some(ctor_hir_id) = struct_def.ctor_id() {
                        this.create_def(ctor_hir_id, DefPathData::Ctor, i.span);
                    }
                }
                _ => {}
            }
            visit::walk_item(this, i);
        });
    }

    fn visit_use_tree(&mut self, use_tree: &'a UseTree, id: NodeId, _nested: bool) {
        self.create_def(id, DefPathData::Misc, use_tree.span);
        visit::walk_use_tree(self, use_tree, id);
    }

    fn visit_foreign_item(&mut self, foreign_item: &'a ForeignItem) {
        if let ForeignItemKind::Macro(_) = foreign_item.node {
            return self.visit_macro_invoc(foreign_item.id);
        }

        let def = self.create_def(foreign_item.id,
                                  DefPathData::ValueNs(foreign_item.ident.as_interned_str()),
                                  foreign_item.span);

        self.with_parent(def, |this| {
            visit::walk_foreign_item(this, foreign_item);
        });
    }

    fn visit_variant(&mut self, v: &'a Variant, g: &'a Generics, item_id: NodeId) {
        let def = self.create_def(v.node.id,
                                  DefPathData::TypeNs(v.node.ident.as_interned_str()),
                                  v.span);
        self.with_parent(def, |this| {
            if let Some(ctor_hir_id) = v.node.data.ctor_id() {
                this.create_def(ctor_hir_id, DefPathData::Ctor, v.span);
            }
            visit::walk_variant(this, v, g, item_id)
        });
    }

    fn visit_variant_data(&mut self, data: &'a VariantData, _: Ident,
                          _: &'a Generics, _: NodeId, _: Span) {
        for (index, field) in data.fields().iter().enumerate() {
            let name = field.ident.map(|ident| ident.name)
                .unwrap_or_else(|| sym::integer(index));
            let def = self.create_def(field.id,
                                      DefPathData::ValueNs(name.as_interned_str()),
                                      field.span);
            self.with_parent(def, |this| this.visit_struct_field(field));
        }
    }

    fn visit_generic_param(&mut self, param: &'a GenericParam) {
        let name = param.ident.as_interned_str();
        let def_path_data = match param.kind {
            GenericParamKind::Lifetime { .. } => DefPathData::LifetimeNs(name),
            GenericParamKind::Type { .. } => DefPathData::TypeNs(name),
            GenericParamKind::Const { .. } => DefPathData::ValueNs(name),
        };
        self.create_def(param.id, def_path_data, param.ident.span);

        visit::walk_generic_param(self, param);
    }

    fn visit_trait_item(&mut self, ti: &'a TraitItem) {
        let def_data = match ti.node {
            TraitItemKind::Method(..) | TraitItemKind::Const(..) =>
                DefPathData::ValueNs(ti.ident.as_interned_str()),
            TraitItemKind::Type(..) => {
                DefPathData::TypeNs(ti.ident.as_interned_str())
            },
            TraitItemKind::Macro(..) => return self.visit_macro_invoc(ti.id),
        };

        let def = self.create_def(ti.id, def_data, ti.span);
        self.with_parent(def, |this| visit::walk_trait_item(this, ti));
    }

    fn visit_impl_item(&mut self, ii: &'a ImplItem) {
        let def_data = match ii.node {
            ImplItemKind::Method(MethodSig {
                ref header,
                ref decl,
            }, ref body) if header.asyncness.node.is_async() => {
                return self.visit_async_fn(
                    ii.id,
                    ii.ident.name,
                    ii.span,
                    header,
                    &ii.generics,
                    decl,
                    body,
                )
            }
            ImplItemKind::Method(..) | ImplItemKind::Const(..) =>
                DefPathData::ValueNs(ii.ident.as_interned_str()),
            ImplItemKind::Type(..) |
            ImplItemKind::Existential(..) => {
                DefPathData::TypeNs(ii.ident.as_interned_str())
            },
            ImplItemKind::Macro(..) => return self.visit_macro_invoc(ii.id),
        };

        let def = self.create_def(ii.id, def_data, ii.span);
        self.with_parent(def, |this| visit::walk_impl_item(this, ii));
    }

    fn visit_pat(&mut self, pat: &'a Pat) {
        match pat.node {
            PatKind::Mac(..) => return self.visit_macro_invoc(pat.id),
            _ => visit::walk_pat(self, pat),
        }
    }

    fn visit_anon_const(&mut self, constant: &'a AnonConst) {
        let def = self.create_def(constant.id,
                                  DefPathData::AnonConst,
                                  constant.value.span);
        self.with_parent(def, |this| visit::walk_anon_const(this, constant));
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        let parent_def = self.parent_def;

        match expr.node {
            ExprKind::Mac(..) => return self.visit_macro_invoc(expr.id),
            ExprKind::Closure(_, asyncness, ..) => {
                let closure_def = self.create_def(expr.id,
                                          DefPathData::ClosureExpr,
                                          expr.span);
                self.parent_def = Some(closure_def);

                // Async closures desugar to closures inside of closures, so
                // we must create two defs.
                if let IsAsync::Async { closure_id, .. } = asyncness {
                    let async_def = self.create_def(closure_id,
                                                    DefPathData::ClosureExpr,
                                                    expr.span);
                    self.parent_def = Some(async_def);
                }
            }
            ExprKind::Async(_, async_id, _) => {
                let async_def = self.create_def(async_id,
                                                DefPathData::ClosureExpr,
                                                expr.span);
                self.parent_def = Some(async_def);
            }
            _ => {}
        };

        visit::walk_expr(self, expr);
        self.parent_def = parent_def;
    }

    fn visit_ty(&mut self, ty: &'a Ty) {
        match ty.node {
            TyKind::Mac(..) => return self.visit_macro_invoc(ty.id),
            TyKind::ImplTrait(node_id, _) => {
                self.create_def(node_id, DefPathData::ImplTrait, ty.span);
            }
            _ => {}
        }
        visit::walk_ty(self, ty);
    }

    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt.node {
            StmtKind::Mac(..) => self.visit_macro_invoc(stmt.id),
            _ => visit::walk_stmt(self, stmt),
        }
    }

    fn visit_token(&mut self, t: Token) {
        if let token::Interpolated(nt) = t.kind {
            if let token::NtExpr(ref expr) = *nt {
                if let ExprKind::Mac(..) = expr.node {
                    self.visit_macro_invoc(expr.id);
                }
            }
        }
    }
}
