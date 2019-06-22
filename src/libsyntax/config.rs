use crate::attr::HasAttrs;
use crate::feature_gate::{
    feature_err,
    EXPLAIN_STMT_ATTR_SYNTAX,
    Features,
    get_features,
    GateIssue,
};
use crate::attr;
use crate::ast;
use crate::edition::Edition;
use crate::mut_visit::*;
use crate::parse::{token, ParseSess};
use crate::ptr::P;
use crate::symbol::sym;
use crate::util::map_in_place::MapInPlace;

use errors::Applicability;
use smallvec::SmallVec;

/// A folder that strips out items that do not belong in the current configuration.
pub struct StripUnconfigured<'a> {
    pub sess: &'a ParseSess,
    pub features: Option<&'a Features>,
}

// `cfg_attr`-process the crate's attributes and compute the crate's features.
pub fn features(mut krate: ast::Crate, sess: &ParseSess, edition: Edition,
                allow_features: &Option<Vec<String>>) -> (ast::Crate, Features) {
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

        features = get_features(&sess.span_diagnostic, &krate.attrs, edition, allow_features);

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
        node.visit_attrs(|attrs| {
            attrs.flat_map_in_place(|attr| self.process_cfg_attr(attr));
        });
    }

    /// Parse and expand a single `cfg_attr` attribute into a list of attributes
    /// when the configuration predicate is true, or otherwise expand into an
    /// empty list of attributes.
    ///
    /// Gives a compiler warning when the `cfg_attr` contains no attributes and
    /// is in the original source file. Gives a compiler error if the syntax of
    /// the attribute is incorrect.
    fn process_cfg_attr(&mut self, attr: ast::Attribute) -> Vec<ast::Attribute> {
        if attr.path != sym::cfg_attr {
            return vec![attr];
        }
        if attr.tokens.is_empty() {
            self.sess.span_diagnostic
                .struct_span_err(
                    attr.span,
                    "malformed `cfg_attr` attribute input",
                ).span_suggestion(
                    attr.span,
                    "missing condition and attribute",
                    "#[cfg_attr(condition, attribute, other_attribute, ...)]".to_owned(),
                    Applicability::HasPlaceholders,
                ).note("for more information, visit \
                       <https://doc.rust-lang.org/reference/conditional-compilation.html\
                       #the-cfg_attr-attribute>")
                .emit();
            return vec![];
        }

        let (cfg_predicate, expanded_attrs) = match attr.parse(self.sess, |parser| {
            parser.expect(&token::OpenDelim(token::Paren))?;

            let cfg_predicate = parser.parse_meta_item()?;
            parser.expect(&token::Comma)?;

            // Presumably, the majority of the time there will only be one attr.
            let mut expanded_attrs = Vec::with_capacity(1);

            while !parser.check(&token::CloseDelim(token::Paren)) {
                let lo = parser.token.span.lo();
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
                return vec![];
            }
        };

        // Lint on zero attributes in source.
        if expanded_attrs.is_empty() {
            return vec![attr];
        }

        // At this point we know the attribute is considered used.
        attr::mark_used(&attr);

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
            vec![]
        }
    }

    /// Determines if a node with the given attributes should be included in this configuration.
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
                return error(nested_meta_items.last().unwrap().span(),
                             "multiple `cfg` predicates are specified", "");
            }

            match nested_meta_items[0].meta_item() {
                Some(meta_item) => attr::cfg_matches(meta_item, self.sess, self.features),
                None => error(nested_meta_items[0].span(),
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
                                      sym::stmt_expr_attributes,
                                      attr.span,
                                      GateIssue::Language,
                                      EXPLAIN_STMT_ATTR_SYNTAX);

            if attr.is_sugared_doc {
                err.help("`///` is for documentation comments. For a plain comment, use `//`.");
            }

            err.emit();
        }
    }

    pub fn configure_foreign_mod(&mut self, foreign_mod: &mut ast::ForeignMod) {
        let ast::ForeignMod { abi: _, items } = foreign_mod;
        items.flat_map_in_place(|item| self.configure(item));
    }

    pub fn configure_generic_params(&mut self, params: &mut Vec<ast::GenericParam>) {
        params.flat_map_in_place(|param| self.configure(param));
    }

    fn configure_variant_data(&mut self, vdata: &mut ast::VariantData) {
        match vdata {
            ast::VariantData::Struct(fields, ..) | ast::VariantData::Tuple(fields, _) =>
                fields.flat_map_in_place(|field| self.configure(field)),
            ast::VariantData::Unit(_) => {}
        }
    }

    pub fn configure_item_kind(&mut self, item: &mut ast::ItemKind) {
        match item {
            ast::ItemKind::Struct(def, _generics) |
            ast::ItemKind::Union(def, _generics) => self.configure_variant_data(def),
            ast::ItemKind::Enum(ast::EnumDef { variants }, _generics) => {
                variants.flat_map_in_place(|variant| self.configure(variant));
                for variant in variants {
                    self.configure_variant_data(&mut variant.node.data);
                }
            }
            _ => {}
        }
    }

    pub fn configure_expr_kind(&mut self, expr_kind: &mut ast::ExprKind) {
        match expr_kind {
            ast::ExprKind::Match(_m, arms) => {
                arms.flat_map_in_place(|arm| self.configure(arm));
            }
            ast::ExprKind::Struct(_path, fields, _base) => {
                fields.flat_map_in_place(|field| self.configure(field));
            }
            _ => {}
        }
    }

    pub fn configure_expr(&mut self, expr: &mut P<ast::Expr>) {
        self.visit_expr_attrs(expr.attrs());

        // If an expr is valid to cfg away it will have been removed by the
        // outer stmt or expression folder before descending in here.
        // Anything else is always required, and thus has to error out
        // in case of a cfg attr.
        //
        // N.B., this is intentionally not part of the visit_expr() function
        //     in order for filter_map_expr() to be able to avoid this check
        if let Some(attr) = expr.attrs().iter().find(|a| is_cfg(a)) {
            let msg = "removing an expression is not supported in this position";
            self.sess.span_diagnostic.span_err(attr.span, msg);
        }

        self.process_cfg_attrs(expr)
    }

    pub fn configure_pat(&mut self, pat: &mut P<ast::Pat>) {
        if let ast::PatKind::Struct(_path, fields, _etc) = &mut pat.node {
            fields.flat_map_in_place(|field| self.configure(field));
        }
    }

    pub fn configure_fn_decl(&mut self, fn_decl: &mut ast::FnDecl) {
        fn_decl.inputs.flat_map_in_place(|arg| self.configure(arg));
    }
}

impl<'a> MutVisitor for StripUnconfigured<'a> {
    fn visit_foreign_mod(&mut self, foreign_mod: &mut ast::ForeignMod) {
        self.configure_foreign_mod(foreign_mod);
        noop_visit_foreign_mod(foreign_mod, self);
    }

    fn visit_item_kind(&mut self, item: &mut ast::ItemKind) {
        self.configure_item_kind(item);
        noop_visit_item_kind(item, self);
    }

    fn visit_expr(&mut self, expr: &mut P<ast::Expr>) {
        self.configure_expr(expr);
        self.configure_expr_kind(&mut expr.node);
        noop_visit_expr(expr, self);
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let mut expr = configure!(self, expr);
        self.configure_expr_kind(&mut expr.node);
        noop_visit_expr(&mut expr, self);
        Some(expr)
    }

    fn flat_map_stmt(&mut self, stmt: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        noop_flat_map_stmt(configure!(self, stmt), self)
    }

    fn flat_map_item(&mut self, item: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        noop_flat_map_item(configure!(self, item), self)
    }

    fn flat_map_impl_item(&mut self, item: ast::ImplItem) -> SmallVec<[ast::ImplItem; 1]> {
        noop_flat_map_impl_item(configure!(self, item), self)
    }

    fn flat_map_trait_item(&mut self, item: ast::TraitItem) -> SmallVec<[ast::TraitItem; 1]> {
        noop_flat_map_trait_item(configure!(self, item), self)
    }

    fn visit_mac(&mut self, _mac: &mut ast::Mac) {
        // Don't configure interpolated AST (cf. issue #34171).
        // Interpolated AST will get configured once the surrounding tokens are parsed.
    }

    fn visit_pat(&mut self, pat: &mut P<ast::Pat>) {
        self.configure_pat(pat);
        noop_visit_pat(pat, self)
    }

    fn visit_fn_decl(&mut self, mut fn_decl: &mut P<ast::FnDecl>) {
        self.configure_fn_decl(&mut fn_decl);
        noop_visit_fn_decl(fn_decl, self);
    }
}

fn is_cfg(attr: &ast::Attribute) -> bool {
    attr.check_name(sym::cfg)
}
