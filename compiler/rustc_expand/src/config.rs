//! Conditional compilation stripping.

use rustc_ast::attr::HasAttrs;
use rustc_ast::mut_visit::*;
use rustc_ast::ptr::P;
use rustc_ast::token::{DelimToken, Token, TokenKind};
use rustc_ast::tokenstream::{DelimSpan, LazyTokenStream, Spacing, TokenStream, TokenTree};
use rustc_ast::{self as ast, AttrItem, Attribute, MetaItem};
use rustc_attr as attr;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::map_in_place::MapInPlace;
use rustc_errors::{error_code, struct_span_err, Applicability, Handler};
use rustc_feature::{Feature, Features, State as FeatureState};
use rustc_feature::{
    ACCEPTED_FEATURES, ACTIVE_FEATURES, REMOVED_FEATURES, STABLE_REMOVED_FEATURES,
};
use rustc_parse::{parse_in, validate_attr};
use rustc_session::parse::feature_err;
use rustc_session::Session;
use rustc_span::edition::{Edition, ALL_EDITIONS};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{Span, DUMMY_SP};

use smallvec::SmallVec;

/// A folder that strips out items that do not belong in the current configuration.
pub struct StripUnconfigured<'a> {
    pub sess: &'a Session,
    pub features: Option<&'a Features>,
}

fn get_features(
    sess: &Session,
    span_handler: &Handler,
    krate_attrs: &[ast::Attribute],
) -> Features {
    fn feature_removed(span_handler: &Handler, span: Span, reason: Option<&str>) {
        let mut err = struct_span_err!(span_handler, span, E0557, "feature has been removed");
        err.span_label(span, "feature has been removed");
        if let Some(reason) = reason {
            err.note(reason);
        }
        err.emit();
    }

    fn active_features_up_to(edition: Edition) -> impl Iterator<Item = &'static Feature> {
        ACTIVE_FEATURES.iter().filter(move |feature| {
            if let Some(feature_edition) = feature.edition {
                feature_edition <= edition
            } else {
                false
            }
        })
    }

    let mut features = Features::default();
    let mut edition_enabled_features = FxHashMap::default();
    let crate_edition = sess.edition();

    for &edition in ALL_EDITIONS {
        if edition <= crate_edition {
            // The `crate_edition` implies its respective umbrella feature-gate
            // (i.e., `#![feature(rust_20XX_preview)]` isn't needed on edition 20XX).
            edition_enabled_features.insert(edition.feature_name(), edition);
        }
    }

    for feature in active_features_up_to(crate_edition) {
        feature.set(&mut features, DUMMY_SP);
        edition_enabled_features.insert(feature.name, crate_edition);
    }

    // Process the edition umbrella feature-gates first, to ensure
    // `edition_enabled_features` is completed before it's queried.
    for attr in krate_attrs {
        if !sess.check_name(attr, sym::feature) {
            continue;
        }

        let list = match attr.meta_item_list() {
            Some(list) => list,
            None => continue,
        };

        for mi in list {
            if !mi.is_word() {
                continue;
            }

            let name = mi.name_or_empty();

            let edition = ALL_EDITIONS.iter().find(|e| name == e.feature_name()).copied();
            if let Some(edition) = edition {
                if edition <= crate_edition {
                    continue;
                }

                for feature in active_features_up_to(edition) {
                    // FIXME(Manishearth) there is currently no way to set
                    // lib features by edition
                    feature.set(&mut features, DUMMY_SP);
                    edition_enabled_features.insert(feature.name, edition);
                }
            }
        }
    }

    for attr in krate_attrs {
        if !sess.check_name(attr, sym::feature) {
            continue;
        }

        let list = match attr.meta_item_list() {
            Some(list) => list,
            None => continue,
        };

        let bad_input = |span| {
            struct_span_err!(span_handler, span, E0556, "malformed `feature` attribute input")
        };

        for mi in list {
            let name = match mi.ident() {
                Some(ident) if mi.is_word() => ident.name,
                Some(ident) => {
                    bad_input(mi.span())
                        .span_suggestion(
                            mi.span(),
                            "expected just one word",
                            format!("{}", ident.name),
                            Applicability::MaybeIncorrect,
                        )
                        .emit();
                    continue;
                }
                None => {
                    bad_input(mi.span()).span_label(mi.span(), "expected just one word").emit();
                    continue;
                }
            };

            if let Some(edition) = edition_enabled_features.get(&name) {
                let msg =
                    &format!("the feature `{}` is included in the Rust {} edition", name, edition);
                span_handler.struct_span_warn_with_code(mi.span(), msg, error_code!(E0705)).emit();
                continue;
            }

            if ALL_EDITIONS.iter().any(|e| name == e.feature_name()) {
                // Handled in the separate loop above.
                continue;
            }

            let removed = REMOVED_FEATURES.iter().find(|f| name == f.name);
            let stable_removed = STABLE_REMOVED_FEATURES.iter().find(|f| name == f.name);
            if let Some(Feature { state, .. }) = removed.or(stable_removed) {
                if let FeatureState::Removed { reason } | FeatureState::Stabilized { reason } =
                    state
                {
                    feature_removed(span_handler, mi.span(), *reason);
                    continue;
                }
            }

            if let Some(Feature { since, .. }) = ACCEPTED_FEATURES.iter().find(|f| name == f.name) {
                let since = Some(Symbol::intern(since));
                features.declared_lang_features.push((name, mi.span(), since));
                continue;
            }

            if let Some(allowed) = sess.opts.debugging_opts.allow_features.as_ref() {
                if allowed.iter().find(|&f| name.as_str() == *f).is_none() {
                    struct_span_err!(
                        span_handler,
                        mi.span(),
                        E0725,
                        "the feature `{}` is not in the list of allowed features",
                        name
                    )
                    .emit();
                    continue;
                }
            }

            if let Some(f) = ACTIVE_FEATURES.iter().find(|f| name == f.name) {
                f.set(&mut features, mi.span());
                features.declared_lang_features.push((name, mi.span(), None));
                continue;
            }

            features.declared_lib_features.push((name, mi.span()));
        }
    }

    features
}

// `cfg_attr`-process the crate's attributes and compute the crate's features.
pub fn features(sess: &Session, mut krate: ast::Crate) -> (ast::Crate, Features) {
    let mut strip_unconfigured = StripUnconfigured { sess, features: None };

    let unconfigured_attrs = krate.attrs.clone();
    let diag = &sess.parse_sess.span_diagnostic;
    let err_count = diag.err_count();
    let features = match strip_unconfigured.configure(krate.attrs) {
        None => {
            // The entire crate is unconfigured.
            krate.attrs = Vec::new();
            krate.module.items = Vec::new();
            Features::default()
        }
        Some(attrs) => {
            krate.attrs = attrs;
            let features = get_features(sess, diag, &krate.attrs);
            if err_count == diag.err_count() {
                // Avoid reconfiguring malformed `cfg_attr`s.
                strip_unconfigured.features = Some(&features);
                strip_unconfigured.configure(unconfigured_attrs);
            }
            features
        }
    };
    (krate, features)
}

#[macro_export]
macro_rules! configure {
    ($this:ident, $node:ident) => {
        match $this.configure($node) {
            Some(node) => node,
            None => return Default::default(),
        }
    };
}

const CFG_ATTR_GRAMMAR_HELP: &str = "#[cfg_attr(condition, attribute, other_attribute, ...)]";
const CFG_ATTR_NOTE_REF: &str = "for more information, visit \
    <https://doc.rust-lang.org/reference/conditional-compilation.html\
    #the-cfg_attr-attribute>";

impl<'a> StripUnconfigured<'a> {
    pub fn configure<T: HasAttrs>(&mut self, mut node: T) -> Option<T> {
        self.process_cfg_attrs(&mut node);
        self.in_cfg(node.attrs()).then_some(node)
    }

    /// Parse and expand all `cfg_attr` attributes into a list of attributes
    /// that are within each `cfg_attr` that has a true configuration predicate.
    ///
    /// Gives compiler warnings if any `cfg_attr` does not contain any
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
    fn process_cfg_attr(&mut self, attr: Attribute) -> Vec<Attribute> {
        if !attr.has_name(sym::cfg_attr) {
            return vec![attr];
        }

        let (cfg_predicate, expanded_attrs) = match self.parse_cfg_attr(&attr) {
            None => return vec![],
            Some(r) => r,
        };

        // Lint on zero attributes in source.
        if expanded_attrs.is_empty() {
            return vec![attr];
        }

        // At this point we know the attribute is considered used.
        self.sess.mark_attr_used(&attr);

        if !attr::cfg_matches(&cfg_predicate, &self.sess.parse_sess, self.features) {
            return vec![];
        }

        // We call `process_cfg_attr` recursively in case there's a
        // `cfg_attr` inside of another `cfg_attr`. E.g.
        //  `#[cfg_attr(false, cfg_attr(true, some_attr))]`.
        expanded_attrs
            .into_iter()
            .flat_map(|(item, span)| {
                let orig_tokens = attr.tokens();

                // We are taking an attribute of the form `#[cfg_attr(pred, attr)]`
                // and producing an attribute of the form `#[attr]`. We
                // have captured tokens for `attr` itself, but we need to
                // synthesize tokens for the wrapper `#` and `[]`, which
                // we do below.

                // Use the `#` in `#[cfg_attr(pred, attr)]` as the `#` token
                // for `attr` when we expand it to `#[attr]`
                let pound_token = orig_tokens.trees().next().unwrap();
                if !matches!(pound_token, TokenTree::Token(Token { kind: TokenKind::Pound, .. })) {
                    panic!("Bad tokens for attribute {:?}", attr);
                }
                // We don't really have a good span to use for the syntheized `[]`
                // in `#[attr]`, so just use the span of the `#` token.
                let bracket_group = TokenTree::Delimited(
                    DelimSpan::from_single(pound_token.span()),
                    DelimToken::Bracket,
                    item.tokens
                        .as_ref()
                        .unwrap_or_else(|| panic!("Missing tokens for {:?}", item))
                        .create_token_stream(),
                );
                let tokens = Some(LazyTokenStream::new(TokenStream::new(vec![
                    (pound_token, Spacing::Alone),
                    (bracket_group, Spacing::Alone),
                ])));

                self.process_cfg_attr(attr::mk_attr_from_item(item, tokens, attr.style, span))
            })
            .collect()
    }

    fn parse_cfg_attr(&self, attr: &Attribute) -> Option<(MetaItem, Vec<(AttrItem, Span)>)> {
        match attr.get_normal_item().args {
            ast::MacArgs::Delimited(dspan, delim, ref tts) if !tts.is_empty() => {
                let msg = "wrong `cfg_attr` delimiters";
                validate_attr::check_meta_bad_delim(&self.sess.parse_sess, dspan, delim, msg);
                match parse_in(&self.sess.parse_sess, tts.clone(), "`cfg_attr` input", |p| {
                    p.parse_cfg_attr()
                }) {
                    Ok(r) => return Some(r),
                    Err(mut e) => {
                        e.help(&format!("the valid syntax is `{}`", CFG_ATTR_GRAMMAR_HELP))
                            .note(CFG_ATTR_NOTE_REF)
                            .emit();
                    }
                }
            }
            _ => self.error_malformed_cfg_attr_missing(attr.span),
        }
        None
    }

    fn error_malformed_cfg_attr_missing(&self, span: Span) {
        self.sess
            .parse_sess
            .span_diagnostic
            .struct_span_err(span, "malformed `cfg_attr` attribute input")
            .span_suggestion(
                span,
                "missing condition and attribute",
                CFG_ATTR_GRAMMAR_HELP.to_string(),
                Applicability::HasPlaceholders,
            )
            .note(CFG_ATTR_NOTE_REF)
            .emit();
    }

    /// Determines if a node with the given attributes should be included in this configuration.
    pub fn in_cfg(&self, attrs: &[Attribute]) -> bool {
        attrs.iter().all(|attr| {
            if !is_cfg(self.sess, attr) {
                return true;
            }
            let meta_item = match validate_attr::parse_meta(&self.sess.parse_sess, attr) {
                Ok(meta_item) => meta_item,
                Err(mut err) => {
                    err.emit();
                    return true;
                }
            };
            let error = |span, msg, suggestion: &str| {
                let mut err = self.sess.parse_sess.span_diagnostic.struct_span_err(span, msg);
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
            let span = meta_item.span;
            match meta_item.meta_item_list() {
                None => error(span, "`cfg` is not followed by parentheses", "cfg(/* predicate */)"),
                Some([]) => error(span, "`cfg` predicate is not specified", ""),
                Some([_, .., l]) => error(l.span(), "multiple `cfg` predicates are specified", ""),
                Some([single]) => match single.meta_item() {
                    Some(meta_item) => {
                        attr::cfg_matches(meta_item, &self.sess.parse_sess, self.features)
                    }
                    None => error(single.span(), "`cfg` predicate key cannot be a literal", ""),
                },
            }
        })
    }

    /// Visit attributes on expression and statements (but not attributes on items in blocks).
    fn visit_expr_attrs(&mut self, attrs: &[Attribute]) {
        // flag the offending attributes
        for attr in attrs.iter() {
            self.maybe_emit_expr_attr_err(attr);
        }
    }

    /// If attributes are not allowed on expressions, emit an error for `attr`
    pub fn maybe_emit_expr_attr_err(&self, attr: &Attribute) {
        if !self.features.map(|features| features.stmt_expr_attributes).unwrap_or(true) {
            let mut err = feature_err(
                &self.sess.parse_sess,
                sym::stmt_expr_attributes,
                attr.span,
                "attributes on expressions are experimental",
            );

            if attr.is_doc_comment() {
                err.help("`///` is for documentation comments. For a plain comment, use `//`.");
            }

            err.emit();
        }
    }

    pub fn configure_foreign_mod(&mut self, foreign_mod: &mut ast::ForeignMod) {
        let ast::ForeignMod { unsafety: _, abi: _, items } = foreign_mod;
        items.flat_map_in_place(|item| self.configure(item));
    }

    fn configure_variant_data(&mut self, vdata: &mut ast::VariantData) {
        match vdata {
            ast::VariantData::Struct(fields, ..) | ast::VariantData::Tuple(fields, _) => {
                fields.flat_map_in_place(|field| self.configure(field))
            }
            ast::VariantData::Unit(_) => {}
        }
    }

    pub fn configure_item_kind(&mut self, item: &mut ast::ItemKind) {
        match item {
            ast::ItemKind::Struct(def, _generics) | ast::ItemKind::Union(def, _generics) => {
                self.configure_variant_data(def)
            }
            ast::ItemKind::Enum(ast::EnumDef { variants }, _generics) => {
                variants.flat_map_in_place(|variant| self.configure(variant));
                for variant in variants {
                    self.configure_variant_data(&mut variant.data);
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
        if let Some(attr) = expr.attrs().iter().find(|a| is_cfg(self.sess, a)) {
            let msg = "removing an expression is not supported in this position";
            self.sess.parse_sess.span_diagnostic.span_err(attr.span, msg);
        }

        self.process_cfg_attrs(expr)
    }

    pub fn configure_pat(&mut self, pat: &mut P<ast::Pat>) {
        if let ast::PatKind::Struct(_path, fields, _etc) = &mut pat.kind {
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
        self.configure_expr_kind(&mut expr.kind);
        noop_visit_expr(expr, self);
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let mut expr = configure!(self, expr);
        self.configure_expr_kind(&mut expr.kind);
        noop_visit_expr(&mut expr, self);
        Some(expr)
    }

    fn flat_map_generic_param(
        &mut self,
        param: ast::GenericParam,
    ) -> SmallVec<[ast::GenericParam; 1]> {
        noop_flat_map_generic_param(configure!(self, param), self)
    }

    fn flat_map_stmt(&mut self, stmt: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        noop_flat_map_stmt(configure!(self, stmt), self)
    }

    fn flat_map_item(&mut self, item: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        noop_flat_map_item(configure!(self, item), self)
    }

    fn flat_map_impl_item(&mut self, item: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        noop_flat_map_assoc_item(configure!(self, item), self)
    }

    fn flat_map_trait_item(&mut self, item: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        noop_flat_map_assoc_item(configure!(self, item), self)
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

fn is_cfg(sess: &Session, attr: &Attribute) -> bool {
    sess.check_name(attr, sym::cfg)
}
