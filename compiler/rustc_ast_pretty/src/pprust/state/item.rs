use crate::pp::Breaks::Inconsistent;
use crate::pprust::state::delimited::IterDelimited;
use crate::pprust::state::{AnnNode, PrintState, State, INDENT_UNIT};

use rustc_ast as ast;
use rustc_ast::GenericBound;
use rustc_ast::ModKind;
use rustc_span::symbol::Ident;

fn visibility_qualified(vis: &ast::Visibility, s: &str) -> String {
    format!("{}{}", State::to_string(|s| s.print_visibility(vis)), s)
}

impl<'a> State<'a> {
    fn print_foreign_mod(&mut self, nmod: &ast::ForeignMod, attrs: &[ast::Attribute]) {
        self.print_inner_attributes(attrs);
        for item in &nmod.items {
            self.print_foreign_item(item);
        }
    }

    pub(crate) fn print_foreign_item(&mut self, item: &ast::ForeignItem) {
        let ast::Item { id, span, ident, ref attrs, ref kind, ref vis, tokens: _ } = *item;
        self.ann.pre(self, AnnNode::SubItem(id));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(span.lo());
        self.print_outer_attributes(attrs);
        match kind {
            ast::ForeignItemKind::Fn(box ast::Fn { defaultness, sig, generics, body }) => {
                self.print_fn_full(sig, ident, generics, vis, *defaultness, body.as_deref(), attrs);
            }
            ast::ForeignItemKind::Static(ty, mutbl, body) => {
                let def = ast::Defaultness::Final;
                self.print_item_const(ident, Some(*mutbl), ty, body.as_deref(), vis, def);
            }
            ast::ForeignItemKind::TyAlias(box ast::TyAlias {
                defaultness,
                generics,
                where_clauses,
                where_predicates_split,
                bounds,
                ty,
            }) => {
                self.print_associated_type(
                    ident,
                    generics,
                    *where_clauses,
                    *where_predicates_split,
                    bounds,
                    ty.as_deref(),
                    vis,
                    *defaultness,
                );
            }
            ast::ForeignItemKind::MacCall(m) => {
                self.print_mac(m);
                if m.args.need_semicolon() {
                    self.word(";");
                }
            }
        }
        self.ann.post(self, AnnNode::SubItem(id))
    }

    fn print_item_const(
        &mut self,
        ident: Ident,
        mutbl: Option<ast::Mutability>,
        ty: &ast::Ty,
        body: Option<&ast::Expr>,
        vis: &ast::Visibility,
        defaultness: ast::Defaultness,
    ) {
        self.head("");
        self.print_visibility(vis);
        self.print_defaultness(defaultness);
        let leading = match mutbl {
            None => "const",
            Some(ast::Mutability::Not) => "static",
            Some(ast::Mutability::Mut) => "static mut",
        };
        self.word_space(leading);
        self.print_ident(ident);
        self.word_space(":");
        self.print_type(ty);
        if body.is_some() {
            self.space();
        }
        self.end(); // end the head-ibox
        if let Some(body) = body {
            self.word_space("=");
            self.print_expr(body);
        }
        self.word(";");
        self.end(); // end the outer cbox
    }

    fn print_associated_type(
        &mut self,
        ident: Ident,
        generics: &ast::Generics,
        where_clauses: (ast::TyAliasWhereClause, ast::TyAliasWhereClause),
        where_predicates_split: usize,
        bounds: &ast::GenericBounds,
        ty: Option<&ast::Ty>,
        vis: &ast::Visibility,
        defaultness: ast::Defaultness,
    ) {
        let (before_predicates, after_predicates) =
            generics.where_clause.predicates.split_at(where_predicates_split);
        self.head("");
        self.print_visibility(vis);
        self.print_defaultness(defaultness);
        self.word_space("type");
        self.print_ident(ident);
        self.print_generic_params(&generics.params);
        if !bounds.is_empty() {
            self.word_nbsp(":");
            self.print_type_bounds(bounds);
        }
        self.print_where_clause_parts(where_clauses.0.0, before_predicates);
        if let Some(ty) = ty {
            self.space();
            self.word_space("=");
            self.print_type(ty);
        }
        self.print_where_clause_parts(where_clauses.1.0, after_predicates);
        self.word(";");
        self.end(); // end inner head-block
        self.end(); // end outer head-block
    }

    /// Pretty-prints an item.
    pub(crate) fn print_item(&mut self, item: &ast::Item) {
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(item.span.lo());
        self.print_outer_attributes(&item.attrs);
        self.ann.pre(self, AnnNode::Item(item));
        match item.kind {
            ast::ItemKind::ExternCrate(orig_name) => {
                self.head(visibility_qualified(&item.vis, "extern crate"));
                if let Some(orig_name) = orig_name {
                    self.print_name(orig_name);
                    self.space();
                    self.word("as");
                    self.space();
                }
                self.print_ident(item.ident);
                self.word(";");
                self.end(); // end inner head-block
                self.end(); // end outer head-block
            }
            ast::ItemKind::Use(ref tree) => {
                self.print_visibility(&item.vis);
                self.word_nbsp("use");
                self.print_use_tree(tree);
                self.word(";");
            }
            ast::ItemKind::Static(ref ty, mutbl, ref body) => {
                let def = ast::Defaultness::Final;
                self.print_item_const(item.ident, Some(mutbl), ty, body.as_deref(), &item.vis, def);
            }
            ast::ItemKind::Const(def, ref ty, ref body) => {
                self.print_item_const(item.ident, None, ty, body.as_deref(), &item.vis, def);
            }
            ast::ItemKind::Fn(box ast::Fn { defaultness, ref sig, ref generics, ref body }) => {
                let body = body.as_deref();
                self.print_fn_full(
                    sig,
                    item.ident,
                    generics,
                    &item.vis,
                    defaultness,
                    body,
                    &item.attrs,
                );
            }
            ast::ItemKind::Mod(unsafety, ref mod_kind) => {
                self.head(Self::to_string(|s| {
                    s.print_visibility(&item.vis);
                    s.print_unsafety(unsafety);
                    s.word("mod");
                }));
                self.print_ident(item.ident);

                match mod_kind {
                    ModKind::Loaded(items, ..) => {
                        self.nbsp();
                        self.bopen();
                        self.print_inner_attributes(&item.attrs);
                        for item in items {
                            self.print_item(item);
                        }
                        let empty = item.attrs.is_empty() && items.is_empty();
                        self.bclose(item.span, empty);
                    }
                    ModKind::Unloaded => {
                        self.word(";");
                        self.end(); // end inner head-block
                        self.end(); // end outer head-block
                    }
                }
            }
            ast::ItemKind::ForeignMod(ref nmod) => {
                self.head(Self::to_string(|s| {
                    s.print_unsafety(nmod.unsafety);
                    s.word("extern");
                }));
                if let Some(abi) = nmod.abi {
                    self.print_literal(&abi.as_lit());
                    self.nbsp();
                }
                self.bopen();
                self.print_foreign_mod(nmod, &item.attrs);
                let empty = item.attrs.is_empty() && nmod.items.is_empty();
                self.bclose(item.span, empty);
            }
            ast::ItemKind::GlobalAsm(ref asm) => {
                self.head(visibility_qualified(&item.vis, "global_asm!"));
                self.print_inline_asm(asm);
                self.word(";");
                self.end();
                self.end();
            }
            ast::ItemKind::TyAlias(box ast::TyAlias {
                defaultness,
                ref generics,
                where_clauses,
                where_predicates_split,
                ref bounds,
                ref ty,
            }) => {
                let ty = ty.as_deref();
                self.print_associated_type(
                    item.ident,
                    generics,
                    where_clauses,
                    where_predicates_split,
                    bounds,
                    ty,
                    &item.vis,
                    defaultness,
                );
            }
            ast::ItemKind::Enum(ref enum_definition, ref params) => {
                self.print_enum_def(enum_definition, params, item.ident, item.span, &item.vis);
            }
            ast::ItemKind::Struct(ref struct_def, ref generics) => {
                self.head(visibility_qualified(&item.vis, "struct"));
                self.print_struct(struct_def, generics, item.ident, item.span, true);
            }
            ast::ItemKind::Union(ref struct_def, ref generics) => {
                self.head(visibility_qualified(&item.vis, "union"));
                self.print_struct(struct_def, generics, item.ident, item.span, true);
            }
            ast::ItemKind::Impl(box ast::Impl {
                unsafety,
                polarity,
                defaultness,
                constness,
                ref generics,
                ref of_trait,
                ref self_ty,
                ref items,
            }) => {
                self.head("");
                self.print_visibility(&item.vis);
                self.print_defaultness(defaultness);
                self.print_unsafety(unsafety);
                self.word("impl");

                if generics.params.is_empty() {
                    self.nbsp();
                } else {
                    self.print_generic_params(&generics.params);
                    self.space();
                }

                self.print_constness(constness);

                if let ast::ImplPolarity::Negative(_) = polarity {
                    self.word("!");
                }

                if let Some(ref t) = *of_trait {
                    self.print_trait_ref(t);
                    self.space();
                    self.word_space("for");
                }

                self.print_type(self_ty);
                self.print_where_clause(&generics.where_clause);

                self.space();
                self.bopen();
                self.print_inner_attributes(&item.attrs);
                for impl_item in items {
                    self.print_assoc_item(impl_item);
                }
                let empty = item.attrs.is_empty() && items.is_empty();
                self.bclose(item.span, empty);
            }
            ast::ItemKind::Trait(box ast::Trait {
                is_auto,
                unsafety,
                ref generics,
                ref bounds,
                ref items,
                ..
            }) => {
                self.head("");
                self.print_visibility(&item.vis);
                self.print_unsafety(unsafety);
                self.print_is_auto(is_auto);
                self.word_nbsp("trait");
                self.print_ident(item.ident);
                self.print_generic_params(&generics.params);
                let mut real_bounds = Vec::with_capacity(bounds.len());
                for b in bounds.iter() {
                    if let GenericBound::Trait(ref ptr, ast::TraitBoundModifier::Maybe) = *b {
                        self.space();
                        self.word_space("for ?");
                        self.print_trait_ref(&ptr.trait_ref);
                    } else {
                        real_bounds.push(b.clone());
                    }
                }
                if !real_bounds.is_empty() {
                    self.word_nbsp(":");
                    self.print_type_bounds(&real_bounds);
                }
                self.print_where_clause(&generics.where_clause);
                self.word(" ");
                self.bopen();
                self.print_inner_attributes(&item.attrs);
                for trait_item in items {
                    self.print_assoc_item(trait_item);
                }
                let empty = item.attrs.is_empty() && items.is_empty();
                self.bclose(item.span, empty);
            }
            ast::ItemKind::TraitAlias(ref generics, ref bounds) => {
                self.head(visibility_qualified(&item.vis, "trait"));
                self.print_ident(item.ident);
                self.print_generic_params(&generics.params);
                let mut real_bounds = Vec::with_capacity(bounds.len());
                // FIXME(durka) this seems to be some quite outdated syntax
                for b in bounds.iter() {
                    if let GenericBound::Trait(ref ptr, ast::TraitBoundModifier::Maybe) = *b {
                        self.space();
                        self.word_space("for ?");
                        self.print_trait_ref(&ptr.trait_ref);
                    } else {
                        real_bounds.push(b.clone());
                    }
                }
                self.nbsp();
                if !real_bounds.is_empty() {
                    self.word_nbsp("=");
                    self.print_type_bounds(&real_bounds);
                }
                self.print_where_clause(&generics.where_clause);
                self.word(";");
                self.end(); // end inner head-block
                self.end(); // end outer head-block
            }
            ast::ItemKind::MacCall(ref mac) => {
                self.print_mac(mac);
                if mac.args.need_semicolon() {
                    self.word(";");
                }
            }
            ast::ItemKind::MacroDef(ref macro_def) => {
                self.print_mac_def(macro_def, &item.ident, item.span, |state| {
                    state.print_visibility(&item.vis)
                });
            }
        }
        self.ann.post(self, AnnNode::Item(item))
    }

    fn print_enum_def(
        &mut self,
        enum_definition: &ast::EnumDef,
        generics: &ast::Generics,
        ident: Ident,
        span: rustc_span::Span,
        visibility: &ast::Visibility,
    ) {
        self.head(visibility_qualified(visibility, "enum"));
        self.print_ident(ident);
        self.print_generic_params(&generics.params);
        self.print_where_clause(&generics.where_clause);
        self.space();
        self.print_variants(&enum_definition.variants, span)
    }

    fn print_variants(&mut self, variants: &[ast::Variant], span: rustc_span::Span) {
        self.bopen();
        for v in variants {
            self.space_if_not_bol();
            self.maybe_print_comment(v.span.lo());
            self.print_outer_attributes(&v.attrs);
            self.ibox(0);
            self.print_variant(v);
            self.word(",");
            self.end();
            self.maybe_print_trailing_comment(v.span, None);
        }
        let empty = variants.is_empty();
        self.bclose(span, empty)
    }

    pub(crate) fn print_visibility(&mut self, vis: &ast::Visibility) {
        match vis.kind {
            ast::VisibilityKind::Public => self.word_nbsp("pub"),
            ast::VisibilityKind::Restricted { ref path, id: _, shorthand } => {
                let path = Self::to_string(|s| s.print_path(path, false, 0));
                if shorthand && (path == "crate" || path == "self" || path == "super") {
                    self.word_nbsp(format!("pub({})", path))
                } else {
                    self.word_nbsp(format!("pub(in {})", path))
                }
            }
            ast::VisibilityKind::Inherited => {}
        }
    }

    fn print_defaultness(&mut self, defaultness: ast::Defaultness) {
        if let ast::Defaultness::Default(_) = defaultness {
            self.word_nbsp("default");
        }
    }

    fn print_record_struct_body(&mut self, fields: &[ast::FieldDef], span: rustc_span::Span) {
        self.nbsp();
        self.bopen();

        let empty = fields.is_empty();
        if !empty {
            self.hardbreak_if_not_bol();

            for field in fields {
                self.hardbreak_if_not_bol();
                self.maybe_print_comment(field.span.lo());
                self.print_outer_attributes(&field.attrs);
                self.print_visibility(&field.vis);
                self.print_ident(field.ident.unwrap());
                self.word_nbsp(":");
                self.print_type(&field.ty);
                self.word(",");
            }
        }

        self.bclose(span, empty);
    }

    fn print_struct(
        &mut self,
        struct_def: &ast::VariantData,
        generics: &ast::Generics,
        ident: Ident,
        span: rustc_span::Span,
        print_finalizer: bool,
    ) {
        self.print_ident(ident);
        self.print_generic_params(&generics.params);
        match struct_def {
            ast::VariantData::Tuple(..) | ast::VariantData::Unit(..) => {
                if let ast::VariantData::Tuple(..) = struct_def {
                    self.popen();
                    self.commasep(Inconsistent, struct_def.fields(), |s, field| {
                        s.maybe_print_comment(field.span.lo());
                        s.print_outer_attributes(&field.attrs);
                        s.print_visibility(&field.vis);
                        s.print_type(&field.ty)
                    });
                    self.pclose();
                }
                self.print_where_clause(&generics.where_clause);
                if print_finalizer {
                    self.word(";");
                }
                self.end();
                self.end(); // Close the outer-box.
            }
            ast::VariantData::Struct(ref fields, ..) => {
                self.print_where_clause(&generics.where_clause);
                self.print_record_struct_body(fields, span);
            }
        }
    }

    pub(crate) fn print_variant(&mut self, v: &ast::Variant) {
        self.head("");
        self.print_visibility(&v.vis);
        let generics = ast::Generics::default();
        self.print_struct(&v.data, &generics, v.ident, v.span, false);
        if let Some(ref d) = v.disr_expr {
            self.space();
            self.word_space("=");
            self.print_expr(&d.value)
        }
    }

    pub(crate) fn print_assoc_item(&mut self, item: &ast::AssocItem) {
        let ast::Item { id, span, ident, ref attrs, ref kind, ref vis, tokens: _ } = *item;
        self.ann.pre(self, AnnNode::SubItem(id));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(span.lo());
        self.print_outer_attributes(attrs);
        match kind {
            ast::AssocItemKind::Fn(box ast::Fn { defaultness, sig, generics, body }) => {
                self.print_fn_full(sig, ident, generics, vis, *defaultness, body.as_deref(), attrs);
            }
            ast::AssocItemKind::Const(def, ty, body) => {
                self.print_item_const(ident, None, ty, body.as_deref(), vis, *def);
            }
            ast::AssocItemKind::TyAlias(box ast::TyAlias {
                defaultness,
                generics,
                where_clauses,
                where_predicates_split,
                bounds,
                ty,
            }) => {
                self.print_associated_type(
                    ident,
                    generics,
                    *where_clauses,
                    *where_predicates_split,
                    bounds,
                    ty.as_deref(),
                    vis,
                    *defaultness,
                );
            }
            ast::AssocItemKind::MacCall(m) => {
                self.print_mac(m);
                if m.args.need_semicolon() {
                    self.word(";");
                }
            }
        }
        self.ann.post(self, AnnNode::SubItem(id))
    }

    fn print_fn_full(
        &mut self,
        sig: &ast::FnSig,
        name: Ident,
        generics: &ast::Generics,
        vis: &ast::Visibility,
        defaultness: ast::Defaultness,
        body: Option<&ast::Block>,
        attrs: &[ast::Attribute],
    ) {
        if body.is_some() {
            self.head("");
        }
        self.print_visibility(vis);
        self.print_defaultness(defaultness);
        self.print_fn(&sig.decl, sig.header, Some(name), generics);
        if let Some(body) = body {
            self.nbsp();
            self.print_block_with_attrs(body, attrs);
        } else {
            self.word(";");
        }
    }

    pub(crate) fn print_fn(
        &mut self,
        decl: &ast::FnDecl,
        header: ast::FnHeader,
        name: Option<Ident>,
        generics: &ast::Generics,
    ) {
        self.print_fn_header_info(header);
        if let Some(name) = name {
            self.nbsp();
            self.print_ident(name);
        }
        self.print_generic_params(&generics.params);
        self.print_fn_params_and_ret(decl, false);
        self.print_where_clause(&generics.where_clause)
    }

    pub(crate) fn print_fn_params_and_ret(&mut self, decl: &ast::FnDecl, is_closure: bool) {
        let (open, close) = if is_closure { ("|", "|") } else { ("(", ")") };
        self.word(open);
        self.commasep(Inconsistent, &decl.inputs, |s, param| s.print_param(param, is_closure));
        self.word(close);
        self.print_fn_ret_ty(&decl.output)
    }

    fn print_where_clause(&mut self, where_clause: &ast::WhereClause) {
        self.print_where_clause_parts(where_clause.has_where_token, &where_clause.predicates);
    }

    pub(crate) fn print_where_clause_parts(
        &mut self,
        has_where_token: bool,
        predicates: &[ast::WherePredicate],
    ) {
        if predicates.is_empty() && !has_where_token {
            return;
        }

        self.space();
        self.word_space("where");

        for (i, predicate) in predicates.iter().enumerate() {
            if i != 0 {
                self.word_space(",");
            }

            self.print_where_predicate(predicate);
        }
    }

    pub fn print_where_predicate(&mut self, predicate: &ast::WherePredicate) {
        match predicate {
            ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate {
                bound_generic_params,
                bounded_ty,
                bounds,
                ..
            }) => {
                self.print_formal_generic_params(bound_generic_params);
                self.print_type(bounded_ty);
                self.word(":");
                if !bounds.is_empty() {
                    self.nbsp();
                    self.print_type_bounds(bounds);
                }
            }
            ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate {
                lifetime,
                bounds,
                ..
            }) => {
                self.print_lifetime(*lifetime);
                self.word(":");
                if !bounds.is_empty() {
                    self.nbsp();
                    self.print_lifetime_bounds(bounds);
                }
            }
            ast::WherePredicate::EqPredicate(ast::WhereEqPredicate { lhs_ty, rhs_ty, .. }) => {
                self.print_type(lhs_ty);
                self.space();
                self.word_space("=");
                self.print_type(rhs_ty);
            }
        }
    }

    fn print_use_tree(&mut self, tree: &ast::UseTree) {
        match tree.kind {
            ast::UseTreeKind::Simple(rename, ..) => {
                self.print_path(&tree.prefix, false, 0);
                if let Some(rename) = rename {
                    self.nbsp();
                    self.word_nbsp("as");
                    self.print_ident(rename);
                }
            }
            ast::UseTreeKind::Glob => {
                if !tree.prefix.segments.is_empty() {
                    self.print_path(&tree.prefix, false, 0);
                    self.word("::");
                }
                self.word("*");
            }
            ast::UseTreeKind::Nested(ref items) => {
                if !tree.prefix.segments.is_empty() {
                    self.print_path(&tree.prefix, false, 0);
                    self.word("::");
                }
                if items.is_empty() {
                    self.word("{}");
                } else if items.len() == 1 {
                    self.print_use_tree(&items[0].0);
                } else {
                    self.cbox(INDENT_UNIT);
                    self.word("{");
                    self.zerobreak();
                    self.ibox(0);
                    for use_tree in items.iter().delimited() {
                        self.print_use_tree(&use_tree.0);
                        if !use_tree.is_last {
                            self.word(",");
                            if let ast::UseTreeKind::Nested(_) = use_tree.0.kind {
                                self.hardbreak();
                            } else {
                                self.space();
                            }
                        }
                    }
                    self.end();
                    self.trailing_comma();
                    self.offset(-INDENT_UNIT);
                    self.word("}");
                    self.end();
                }
            }
        }
    }
}
