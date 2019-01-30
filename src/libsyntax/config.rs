use attr::HasAttrs;
use feature_gate::{
    feature_err,
    EXPLAIN_STMT_ATTR_SYNTAX,
    Features,
    get_features,
    GateIssue,
};
use attr;
use ast;
use source_map::Spanned;
use edition::Edition;
use parse::{token, ParseSess};
use smallvec::SmallVec;
use errors::Applicability;
use visit_mut::{self, Action};

use ptr::P;

/// A folder that strips out items that do not belong in the current configuration.
pub struct StripUnconfigured<'a> {
    pub sess: &'a ParseSess,
    pub features: Option<&'a Features>,
}

// `cfg_attr`-process the crate's attributes and compute the crate's features.
pub fn features(mut krate: ast::Crate, sess: &ParseSess, edition: Edition)
                -> (ast::Crate, Features) {
    let features;
    {
        let mut strip_unconfigured = StripUnconfigured {
            sess,
            features: None,
        };

        let unconfigured_attrs = krate.attrs.clone();
        let err_count = sess.span_diagnostic.err_count();
        if let Some(attrs) = strip_unconfigured.configure(krate.attrs) {
            krate.attrs = attrs;
        } else { // the entire crate is unconfigured
            krate.attrs = Vec::new();
            krate.module.items = Vec::new();
            return (krate, Features::new());
        }

        features = get_features(&sess.span_diagnostic, &krate.attrs, edition);

        // Avoid reconfiguring malformed `cfg_attr`s
        if err_count == sess.span_diagnostic.err_count() {
            strip_unconfigured.features = Some(&features);
            strip_unconfigured.configure(unconfigured_attrs);
        }
    }

    (krate, features)
}

macro_rules! configure {
    ($this:ident, $node:ident) => {
        match $this.configure($node) {
            Some(node) => node,
            None => return Default::default(),
        }
    }
}

impl<'a> StripUnconfigured<'a> {
    pub fn keep_then<T: HasAttrs, R>(
        &mut self,
        node: &mut T,
        then: impl FnOnce(&mut Self, &mut T)
    ) -> Action<R> {
        if likely!(self.keep(node)) {
            then(self, node);
            Action::Reuse
        } else {
            Action::Remove
        }
    }

    pub fn keep<T: HasAttrs>(&mut self, node: &mut T) -> bool {
        self.process_cfg_attrs(node);
        self.in_cfg(node.attrs())
    }

    pub fn configure<T: HasAttrs>(&mut self, mut node: T) -> Option<T> {
        self.process_cfg_attrs(&mut node);
        if self.in_cfg(node.attrs()) { Some(node) } else { None }
    }

    /// Parse and expand all `cfg_attr` attributes into a list of attributes
    /// that are within each `cfg_attr` that has a true configuration predicate.
    ///
    /// Gives compiler warnigns if any `cfg_attr` does not contain any
    /// attributes and is in the original source code. Gives compiler errors if
    /// the syntax of any `cfg_attr` is incorrect.
    pub fn process_cfg_attrs<T: HasAttrs>(&mut self, node: &mut T) {
        let attrs = if let Some(attrs) = node.attrs_mut() {
            attrs
        } else {
            // No attributes so nothing to do
            return;
        };
        if likely!(attrs.is_empty()) {
            return;
        }
        if likely!(!attrs.iter().any(|attr| attr.check_name("cfg_attr"))) {
            return;
        }
        let new_attrs: Vec<_> = attrs.drain(..).flat_map(|attr| {
            self.process_cfg_attr(attr)
        }).collect();
        *attrs = new_attrs;
    }

    /// Parse and expand a single `cfg_attr` attribute into a list of attributes
    /// when the configuration predicate is true, or otherwise expand into an
    /// empty list of attributes.
    ///
    /// Gives a compiler warning when the `cfg_attr` contains no attributes and
    /// is in the original source file. Gives a compiler error if the syntax of
    /// the attribute is incorrect
    fn process_cfg_attr(&mut self, attr: ast::Attribute) -> SmallVec<[ast::Attribute; 1]> {
        if !attr.check_name("cfg_attr") {
            return smallvec![attr];
        }

        let (cfg_predicate, expanded_attrs) = match attr.parse(self.sess, |parser| {
            parser.expect(&token::OpenDelim(token::Paren))?;

            let cfg_predicate = parser.parse_meta_item()?;
            parser.expect(&token::Comma)?;

            // Presumably, the majority of the time there will only be one attr.
            let mut expanded_attrs: SmallVec<[_; 1]> = SmallVec::new();

            while !parser.check(&token::CloseDelim(token::Paren)) {
                let lo = parser.span.lo();
                let (path, tokens) = parser.parse_meta_item_unrestricted()?;
                expanded_attrs.push((path, tokens, parser.prev_span.with_lo(lo)));
                parser.expect_one_of(&[token::Comma], &[token::CloseDelim(token::Paren)])?;
            }

            parser.expect(&token::CloseDelim(token::Paren))?;
            Ok((cfg_predicate, expanded_attrs))
        }) {
            Ok(result) => result,
            Err(mut e) => {
                e.emit();
                return SmallVec::new();
            }
        };

        // Check feature gate and lint on zero attributes in source. Even if the feature is gated,
        // we still compute as if it wasn't, since the emitted error will stop compilation further
        // along the compilation.
        if expanded_attrs.len() == 0 {
            // FIXME: Emit unused attribute lint here.
        }

        if attr::cfg_matches(&cfg_predicate, self.sess, self.features) {
            // We call `process_cfg_attr` recursively in case there's a
            // `cfg_attr` inside of another `cfg_attr`. E.g.
            //  `#[cfg_attr(false, cfg_attr(true, some_attr))]`.
            expanded_attrs.into_iter()
            .flat_map(|(path, tokens, span)| self.process_cfg_attr(ast::Attribute {
                id: attr::mk_attr_id(),
                style: attr.style,
                path,
                tokens,
                is_sugared_doc: false,
                span,
            }))
            .collect()
        } else {
            SmallVec::new()
        }
    }

    /// Determine if a node with the given attributes should be included in this configuration.
    pub fn in_cfg(&mut self, attrs: &[ast::Attribute]) -> bool {
        attrs.iter().all(|attr| {
            if !is_cfg(attr) {
                return true;
            }

            let error = |span, msg, suggestion: &str| {
                let mut err = self.sess.span_diagnostic.struct_span_err(span, msg);
                if !suggestion.is_empty() {
                    err.span_suggestion(
                        span,
                        "expected syntax is",
                        suggestion.into(),
                        Applicability::MaybeIncorrect,
                    );
                }
                err.emit();
                true
            };

            let meta_item = match attr.parse_meta(self.sess) {
                Ok(meta_item) => meta_item,
                Err(mut err) => { err.emit(); return true; }
            };
            let nested_meta_items = if let Some(nested_meta_items) = meta_item.meta_item_list() {
                nested_meta_items
            } else {
                return error(meta_item.span, "`cfg` is not followed by parentheses",
                                             "cfg(/* predicate */)");
            };

            if nested_meta_items.is_empty() {
                return error(meta_item.span, "`cfg` predicate is not specified", "");
            } else if nested_meta_items.len() > 1 {
                return error(nested_meta_items.last().unwrap().span,
                             "multiple `cfg` predicates are specified", "");
            }

            match nested_meta_items[0].meta_item() {
                Some(meta_item) => attr::cfg_matches(meta_item, self.sess, self.features),
                None => error(nested_meta_items[0].span,
                              "`cfg` predicate key cannot be a literal", ""),
            }
        })
    }

    /// Visit attributes on expression and statements (but not attributes on items in blocks).
    fn visit_expr_attrs(&mut self, attrs: &[ast::Attribute]) {
        // flag the offending attributes
        for attr in attrs.iter() {
            self.maybe_emit_expr_attr_err(attr);
        }
    }

    /// If attributes are not allowed on expressions, emit an error for `attr`
    pub fn maybe_emit_expr_attr_err(&self, attr: &ast::Attribute) {
        if !self.features.map(|features| features.stmt_expr_attributes).unwrap_or(true) {
            let mut err = feature_err(self.sess,
                                      "stmt_expr_attributes",
                                      attr.span,
                                      GateIssue::Language,
                                      EXPLAIN_STMT_ATTR_SYNTAX);

            if attr.is_sugared_doc {
                err.help("`///` is for documentation comments. For a plain comment, use `//`.");
            }

            err.emit();
        }
    }

    pub fn configure_foreign_mod(&mut self, foreign_mod: ast::ForeignMod) -> ast::ForeignMod {
        ast::ForeignMod {
            abi: foreign_mod.abi,
            items: foreign_mod.items.into_iter().filter_map(|item| self.configure(item)).collect(),
        }
    }

    fn configure_variant_data(&mut self, vdata: ast::VariantData) -> ast::VariantData {
        match vdata {
            ast::VariantData::Struct(fields, id) => {
                let fields = fields.into_iter().filter_map(|field| self.configure(field));
                ast::VariantData::Struct(fields.collect(), id)
            }
            ast::VariantData::Tuple(fields, id) => {
                let fields = fields.into_iter().filter_map(|field| self.configure(field));
                ast::VariantData::Tuple(fields.collect(), id)
            }
            ast::VariantData::Unit(id) => ast::VariantData::Unit(id)
        }
    }

    pub fn configure_item_kind(&mut self, item: ast::ItemKind) -> ast::ItemKind {
        match item {
            ast::ItemKind::Struct(def, generics) => {
                ast::ItemKind::Struct(self.configure_variant_data(def), generics)
            }
            ast::ItemKind::Union(def, generics) => {
                ast::ItemKind::Union(self.configure_variant_data(def), generics)
            }
            ast::ItemKind::Enum(def, generics) => {
                let variants = def.variants.into_iter().filter_map(|v| {
                    self.configure(v).map(|v| {
                        Spanned {
                            node: ast::Variant_ {
                                ident: v.node.ident,
                                attrs: v.node.attrs,
                                data: self.configure_variant_data(v.node.data),
                                disr_expr: v.node.disr_expr,
                            },
                            span: v.span
                        }
                    })
                });
                ast::ItemKind::Enum(ast::EnumDef {
                    variants: variants.collect(),
                }, generics)
            }
            item => item,
        }
    }

    pub fn configure_expr_kind(&mut self, expr_kind: ast::ExprKind) -> ast::ExprKind {
        match expr_kind {
            ast::ExprKind::Match(m, arms) => {
                let arms = arms.into_iter().filter_map(|a| self.configure(a)).collect();
                ast::ExprKind::Match(m, arms)
            }
            ast::ExprKind::Struct(path, fields, base) => {
                let fields = fields.into_iter()
                    .filter_map(|field| {
                        self.configure(field)
                    })
                    .collect();
                ast::ExprKind::Struct(path, fields, base)
            }
            _ => expr_kind,
        }
    }

    pub fn configure_expr(&mut self, expr: &mut ast::Expr)  {
        self.visit_expr_attrs(expr.attrs());

        // If an expr is valid to cfg away it will have been removed by the
        // outer stmt or expression folder before descending in here.
        // Anything else is always required, and thus has to error out
        // in case of a cfg attr.
        //
        // N.B., this is intentionally not part of the fold_expr() function
        //     in order for fold_opt_expr() to be able to avoid this check
        if let Some(attr) = expr.attrs().iter().find(|a| is_cfg(a)) {
            let msg = "removing an expression is not supported in this position";
            self.sess.span_diagnostic.span_err(attr.span, msg);
        }
        self.process_cfg_attrs(expr)
    }

    pub fn configure_stmt(&mut self, stmt: ast::Stmt) -> Option<ast::Stmt> {
        self.configure(stmt)
    }

    pub fn configure_struct_expr_field(&mut self, field: ast::Field) -> Option<ast::Field> {
        self.configure(field)
    }

    pub fn configure_pat(&mut self, pattern: P<ast::Pat>) -> P<ast::Pat> {
        pattern.map(|mut pattern| {
            if let ast::PatKind::Struct(path, fields, etc) = pattern.node {
                let fields = fields.into_iter()
                    .filter_map(|field| {
                        self.configure(field)
                    })
                    .collect();
                pattern.node = ast::PatKind::Struct(path, fields, etc);
            }
            pattern
        })
    }

    // deny #[cfg] on generic parameters until we decide what to do with it.
    // see issue #51279.
    pub fn disallow_cfg_on_generic_param(&mut self, param: &ast::GenericParam) {
        for attr in param.attrs() {
            let offending_attr = if attr.check_name("cfg") {
                "cfg"
            } else if attr.check_name("cfg_attr") {
                "cfg_attr"
            } else {
                continue;
            };
            let msg = format!("#[{}] cannot be applied on a generic parameter", offending_attr);
            self.sess.span_diagnostic.span_err(attr.span, &msg);
        }
    }
}

impl<'a> visit_mut::MutVisitor for StripUnconfigured<'a> {
    fn visit_foreign_item(&mut self, i: &mut ast::ForeignItem) -> Action<ast::ForeignItem> {
        self.keep_then(i, |this, i| visit_mut::walk_foreign_item(this, i))
    }

    fn visit_variant(
        &mut self,
        v: &mut ast::Variant,
        g: &mut ast::Generics,
        item_id: ast::NodeId
    ) -> Action<ast::Variant> {
        if likely!(self.keep(v)) {
            visit_mut::walk_variant(self, v, g, item_id);
            Action::Reuse
        } else {
            Action::Remove
        }
    }

    fn visit_struct_field(&mut self, s: &mut ast::StructField) -> Action<ast::StructField> {
        self.keep_then(s, |this, s| visit_mut::walk_struct_field(this, s))
    }

    fn visit_arm(&mut self, a: &mut ast::Arm) -> Action<ast::Arm> {
        self.keep_then(a, |this, a| visit_mut::walk_arm(this, a))
    }

    fn visit_field(&mut self, field: &mut ast::Field) -> Action<ast::Field> {
        self.keep_then(field, |this, field| visit_mut::walk_field(this, field))
    }

    fn visit_opt_expr(&mut self, ex: &mut ast::Expr) -> bool {
        if likely!(self.keep(ex)) {
            visit_mut::walk_expr(self, ex);
            true
        } else {
            false
        }
    }

    fn visit_expr(&mut self, ex: &mut ast::Expr) {
        self.configure_expr(ex);
        visit_mut::walk_expr(self, ex)
    }

    fn visit_stmt(&mut self, s: &mut ast::Stmt) -> Action<ast::Stmt> {
        if likely!(self.keep(s)) {
            visit_mut::walk_stmt(self, s)
        } else {
            Action::Remove
        }
    }

    fn visit_item(&mut self, i: &mut P<ast::Item>) -> Action<P<ast::Item>> {
        self.keep_then(i, |this, i| visit_mut::walk_item(this, i))
    }

    fn visit_trait_item(&mut self, i: &mut ast::TraitItem) -> Action<ast::TraitItem> {
        self.keep_then(i, |this, i| visit_mut::walk_trait_item(this, i))
    }

    fn visit_impl_item(&mut self, ii: &mut ast::ImplItem) -> Action<ast::ImplItem> {
        self.keep_then(ii, |this, ii| visit_mut::walk_impl_item(this, ii))
    }

    fn visit_mac(&mut self, _mac: &mut ast::Mac) {
        // Don't configure interpolated AST (cf. issue #34171).
        // Interpolated AST will get configured once the surrounding tokens are parsed.
    }

    fn visit_field_pat(
        &mut self,
        p: &mut Spanned<ast::FieldPat>
    ) -> Action<Spanned<ast::FieldPat>> {
        self.keep_then(p, |this, p| visit_mut::walk_field_pat(this, p))
    }
}

fn is_cfg(attr: &ast::Attribute) -> bool {
    attr.check_name("cfg")
}
