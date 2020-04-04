use ra_syntax::ast as ra;
#[allow(unused_imports)]
use ra_syntax::ast::{
    ArgListOwner, AstElement, AstNode, AstToken, AttrsOwner, LoopBodyOwner, NameOwner,
    TypeAscriptionOwner, TypeBoundsOwner, TypeParamsOwner, VisibilityOwner,
};
use ra_syntax::{
    NodeOrToken, Parse, SmolStr, SourceFile as RaSourceFile, SyntaxElement, SyntaxToken, TextRange,
    TextUnit,
};
use rustc_ast::ast;
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::tokenstream::{IsJoint, TokenStream};
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Diagnostic, DiagnosticBuilder, FatalError, Level, PResult};
use rustc_parse::lexer;
use rustc_session::parse::ParseSess;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{kw, Symbol};
use rustc_span::{BytePos, FileName, Pos, SourceFile, Span, SyntaxContext};

use std::convert::{TryFrom, TryInto};
use std::path::Path;

pub fn parse_crate_from_file<'tcx>(
    input: &Path,
    sess: &'tcx ParseSess,
) -> PResult<'tcx, ast::Crate> {
    parse_crate_from_source_file(file_to_source_file(sess, input, None), sess)
}

pub fn parse_crate_from_source_str(
    name: FileName,
    source: String,
    sess: &ParseSess,
) -> PResult<'_, ast::Crate> {
    parse_crate_from_source_file(sess.source_map().new_source_file(name, source), sess)
}

/// Given a session and a path and an optional span (for error reporting),
/// add the path to the session's source_map and return the new source_file or
/// error when a file can't be read.
fn try_file_to_source_file(
    sess: &ParseSess,
    path: &Path,
    spanopt: Option<Span>,
) -> Result<Lrc<SourceFile>, Diagnostic> {
    sess.source_map().load_file(path).map_err(|e| {
        let msg = format!("couldn't read {}: {}", path.display(), e);
        let mut diag = Diagnostic::new(Level::Fatal, &msg);
        if let Some(sp) = spanopt {
            diag.set_span(sp);
        }
        diag
    })
}

/// Given a session and a path and an optional span (for error reporting),
/// adds the path to the session's `source_map` and returns the new `source_file`.
fn file_to_source_file(sess: &ParseSess, path: &Path, spanopt: Option<Span>) -> Lrc<SourceFile> {
    match try_file_to_source_file(sess, path, spanopt) {
        Ok(source_file) => source_file,
        Err(d) => {
            sess.span_diagnostic.emit_diagnostic(&d);
            FatalError.raise();
        }
    }
}

fn parse_crate_from_source_file<'tcx>(
    source_file: Lrc<SourceFile>,
    sess: &'tcx ParseSess,
) -> PResult<'tcx, ast::Crate> {
    let ctxt = SyntaxContext::root();
    let src = match source_file.src.as_ref() {
        Some(src) => src,
        None => {
            sess.span_diagnostic
                .bug(&format!("cannot lex `source_file` without source: {}", source_file.name));
        }
    };
    let parse = RaSourceFile::parse(src);
    let parser = Parser::new(source_file, ctxt, sess);
    let krate = parser.to_crate(parse);

    // we recover from all errors after reporting them as diagnostics
    Ok(krate)
}

fn raw_text_range(element: &impl ra::AstElement) -> TextRange {
    match element.syntax_element() {
        NodeOrToken::Node(node) => node.text_range(),
        NodeOrToken::Token(token) => token.text_range(),
    }
}

// TODO: maybe we should move those helpers to rust-analyzer, although it's also possible they are only useful here

/// whitespace and non-doc comments
fn is_ignorable_token(token: &SyntaxToken) -> bool {
    if ra::Whitespace::can_cast(token.kind()) {
        true
    } else if ra::Comment::can_cast(token.kind()) {
        ra::Comment::cast(token.clone()).unwrap().kind().doc.is_none()
    } else {
        false
    }
}

/// whitespace, non-doc and outer doc comments and outer attributes
fn is_outer_element(element: &SyntaxElement) -> bool {
    match element {
        NodeOrToken::Node(node) => {
            ra::Attr::can_cast(node.kind())
                && ra::Attr::cast(node.clone()).unwrap().excl().is_none()
        }
        NodeOrToken::Token(token) => is_outer_token(token),
    }
}

fn is_outer_token(token: &SyntaxToken) -> bool {
    if ra::Whitespace::can_cast(token.kind()) {
        true
    } else if ra::Comment::can_cast(token.kind()) {
        ra::Comment::cast(token.clone()).unwrap().kind().doc != Some(ra::CommentPlacement::Inner)
    } else {
        false
    }
}

/// start of first non-ignorable token
fn outer_start(element: &impl ra::AstElement) -> TextUnit {
    match element.syntax_element() {
        NodeOrToken::Token(token) => token.text_range().start(),
        NodeOrToken::Node(node) => {
            // TODO: we might not need this, if the rust-analyzer parser never adds whitespace/comments at the start of items
            if let Some(mut token) = node.first_token() {
                loop {
                    if !is_ignorable_token(&token) {
                        break token.text_range().start();
                    } else if let Some(token_to_move_to) = token.next_token() {
                        token = token_to_move_to;
                    } else {
                        break token.text_range().end();
                    }
                }
            } else {
                node.text_range().start()
            }
        }
    }
}

/// end of last non-ignorable token
fn outer_end(element: &impl ra::AstElement) -> TextUnit {
    match element.syntax_element() {
        NodeOrToken::Token(token) => token.text_range().end(),
        NodeOrToken::Node(node) => {
            // TODO: we might not need this, except for SourceFile, if the rust-analyzer parser never adds whitespace/comments at the end of items
            if let Some(mut token) = node.last_token() {
                loop {
                    if !is_ignorable_token(&token) {
                        break token.text_range().end();
                    } else if let Some(token_to_move_to) = token.prev_token() {
                        token = token_to_move_to;
                    } else {
                        break token.text_range().start();
                    }
                }
            } else {
                node.text_range().end()
            }
        }
    }
}

/// trims whitespace and comments, includes all attributes and doc comments
fn outer_range(element: &impl ra::AstElement) -> TextRange {
    TextRange::from_to(outer_start(element), outer_end(element))
}

/// start of first non-outer token
fn inner_start(element: &impl ra::AstElement) -> TextUnit {
    match element.syntax_element() {
        NodeOrToken::Token(token) => token.text_range().start(),
        NodeOrToken::Node(node) => {
            if let Some(element) =
                node.children_with_tokens().skip_while(|element| is_outer_element(element)).next()
            {
                match element {
                    NodeOrToken::Token(token) => token.text_range().start(),
                    NodeOrToken::Node(node) => {
                        // TODO: we might not need this, if the rust-analyzer parser never adds whitespace/comments at the start of items
                        if let Some(mut token) = node.first_token() {
                            loop {
                                if !is_outer_token(&token) {
                                    break token.text_range().start();
                                } else if let Some(token_to_move_to) = token.next_token() {
                                    token = token_to_move_to;
                                } else {
                                    break token.text_range().end();
                                }
                            }
                        } else {
                            node.text_range().start()
                        }
                    }
                }
            } else {
                node.text_range().end()
            }
        }
    }
}

/// end of last non-outer token
fn inner_end(element: &impl ra::AstElement) -> TextUnit {
    match element.syntax_element() {
        NodeOrToken::Token(token) => token.text_range().end(),
        NodeOrToken::Node(node) => {
            // TODO: we might not need this, except for SourceFile, if the rust-analyzer parser never adds whitespace/comments at the end of items
            if let Some(mut token) = node.last_token() {
                loop {
                    if !is_outer_token(&token) {
                        break token.text_range().end();
                    } else if let Some(token_to_move_to) = token.prev_token() {
                        token = token_to_move_to;
                    } else {
                        break token.text_range().start();
                    }
                }
            } else {
                node.text_range().end()
            }
        }
    }
}

/// end of last non-outer token
fn inner_last_start(element: &impl ra::AstElement) -> TextUnit {
    match element.syntax_element() {
        NodeOrToken::Token(token) => token.text_range().start(),
        NodeOrToken::Node(node) => {
            // TODO: we might not need this, except for SourceFile, if the rust-analyzer parser never adds whitespace/comments at the end of items
            if let Some(mut token) = node.last_token() {
                loop {
                    if !is_outer_token(&token) {
                        break token.text_range().start();
                    } else if let Some(token_to_move_to) = token.prev_token() {
                        token = token_to_move_to;
                    } else {
                        break node.text_range().start();
                    }
                }
            } else {
                node.text_range().start()
            }
        }
    }
}

/// trims whitespace, comments, outer attributes and doc comments, includes inner attributes and inner doc comments
fn inner_range(element: &impl ra::AstElement) -> TextRange {
    TextRange::from_to(inner_start(element), inner_end(element))
}

/// start of first following non-ignorable token
fn start_of_next(element: &impl ra::AstElement) -> TextUnit {
    if let Some(mut token) = match element.syntax_element() {
        NodeOrToken::Node(node) => node.last_token().and_then(|token| token.next_token()),
        NodeOrToken::Token(token) => token.next_token(),
    } {
        loop {
            if !is_ignorable_token(&token) {
                break token.text_range().start();
            } else if let Some(next) = token.next_token() {
                token = next;
            } else {
                break token.text_range().end();
            }
        }
    } else {
        raw_text_range(element).end()
    }
}

fn end_of_next(element: &impl ra::AstElement) -> TextUnit {
    if let Some(mut token) = match element.syntax_element() {
        NodeOrToken::Node(node) => node.last_token().and_then(|token| token.next_token()),
        NodeOrToken::Token(token) => token.next_token(),
    } {
        loop {
            if !is_ignorable_token(&token) {
                break token.text_range().end();
            } else if let Some(next) = token.next_token() {
                token = next;
            } else {
                break token.text_range().end();
            }
        }
    } else {
        raw_text_range(element).end()
    }
}

/// end of preceding non-ignorable token
fn end_of_prev(element: &impl ra::AstElement) -> TextUnit {
    if let Some(mut token) = match element.syntax_element() {
        NodeOrToken::Node(node) => node.first_token().and_then(|token| token.prev_token()),
        NodeOrToken::Token(token) => token.prev_token(),
    } {
        loop {
            if !is_ignorable_token(&token) {
                break token.text_range().end();
            } else if let Some(prev) = token.prev_token() {
                token = prev;
            } else {
                break token.text_range().start();
            }
        }
    } else {
        raw_text_range(element).start()
    }
}

fn empty_text_range(unit: TextUnit) -> TextRange {
    TextRange::from_to(unit, unit)
}

fn remove_joint(stream: &mut TokenStream) {
    for tree in Lrc::make_mut(&mut stream.0).iter_mut() {
        tree.1 = IsJoint::NonJoint;
        /*
        match &mut tree.0 {
            TokenTree::Delimited(_, _, sub_stream) => remove_joint(sub_stream),
            _ => {}
        }
        */
    }
}

struct Parser<'tcx> {
    file: Lrc<SourceFile>,
    ctxt: SyntaxContext,
    sess: &'tcx ParseSess,
}

#[derive(Debug, Clone, Copy)]
struct FnParamMode {
    require_pat: bool,
    require_type: bool,
}

impl<'tcx> Parser<'tcx> {
    pub fn new(file: Lrc<SourceFile>, ctxt: SyntaxContext, sess: &'tcx ParseSess) -> Self {
        Self { file, ctxt, sess }
    }

    pub fn to_crate(&self, parse: Parse<RaSourceFile>) -> ast::Crate {
        for error in parse.errors() {
            let msg = error.to_string();
            self.error_at_span(self.to_span(error.range()), &msg);
        }

        self.to_crate_from_source_file(parse.tree())
    }

    fn error_at_span(&self, span: Span, msg: &str) {
        let mut diag = Diagnostic::new(Level::Error, msg);
        diag.set_span(span);
        self.sess.span_diagnostic.emit_diagnostic(&diag);
    }

    fn error(&self, element: &impl ra::AstElement, msg: &str) {
        self.error_at_span(self.get_inner_span(element), msg)
    }

    fn missing_in(&self, element: &impl ra::AstElement, object: &str, container: &str) {
        self.error(element, &format!("missing {} in {}", object, container))
    }

    fn invalid_token(&self, token: Option<impl ra::AstToken>, container: &str) {
        if let Some(token) = token {
            self.error_at_span(
                self.get_inner_span(&token),
                &format!("{} token is not valid in {}", token.text(), container),
            )
        }
    }

    fn invalid_element(
        &self,
        element: Option<impl ra::AstElement>,
        _parent: &impl ra::AstElement,
        object: &str,
        container: &str,
    ) {
        if let Some(node) = element {
            self.error_at_span(
                self.get_inner_span(&node),
                &format!("{} is not valid in {}", object, container),
            )
        }
    }

    fn expect_in<T>(
        &self,
        opt: Option<T>,
        element: &impl ra::AstElement,
        object: &str,
        container: &str,
    ) -> Option<T> {
        if opt.is_none() {
            self.missing_in(element, object, container);
        }
        opt
    }

    fn expect_in_res<T>(
        &self,
        res: Result<T, impl ra::AstElement>,
        object: &str,
        container: &str,
    ) -> Option<T> {
        match res {
            Ok(node) => Some(node),
            Err(node) => {
                self.error(&node, &format!("missing {} in {}", object, container));
                None
            }
        }
    }

    fn to_byte_pos(&self, unit: TextUnit) -> BytePos {
        BytePos::from_usize(self.file.start_pos.to_usize() + unit.to_usize())
    }

    fn to_span(&self, range: TextRange) -> Span {
        Span::new(self.to_byte_pos(range.start()), self.to_byte_pos(range.end()), self.ctxt)
    }

    fn to_span_from_unit(&self, unit: TextUnit) -> Span {
        let byte_pos = self.to_byte_pos(unit);
        Span::new(byte_pos, byte_pos, self.ctxt)
    }

    fn get_inner_span(&self, element: &impl ra::AstElement) -> Span {
        self.to_span(inner_range(element))
    }

    fn get_inner_to_next_span(&self, element: &impl ra::AstElement) -> Span {
        self.to_span(TextRange::from_to(inner_start(element), end_of_next(element)))
    }

    fn get_last_token_span(&self, element: &impl ra::AstElement) -> Span {
        self.to_span(TextRange::from_to(inner_last_start(element), inner_end(element)))
    }

    fn to_crate_from_source_file(&self, source_file: RaSourceFile) -> ast::Crate {
        let span = self.to_span(outer_range(&source_file));
        ast::Crate {
            span,
            attrs: self.get_attributes(&source_file).collect(),
            module: ast::Mod {
                inner: span,
                items: self.to_items(Some(source_file), "crate"),
                inline: true,
            },
            proc_macros: Default::default(), // TODO: proc macros
        }
    }

    fn to_defaultness(&self, default_kw: Option<ra::DefaultKw>) -> ast::Defaultness {
        if let Some(default_token) = default_kw {
            ast::Defaultness::Default(self.get_inner_span(&default_token))
        } else {
            ast::Defaultness::Final
        }
    }

    fn to_constness(&self, const_kw: Option<ra::ConstKw>) -> ast::Const {
        if let Some(const_token) = const_kw {
            ast::Const::Yes(self.get_inner_span(&const_token))
        } else {
            ast::Const::No
        }
    }

    fn get_visibility(
        &self,
        node: &(impl VisibilityOwner + AstElement),
    ) -> Spanned<ast::VisibilityKind> {
        self.to_visibility(node.visibility(), node)
    }

    fn to_visibility(
        &self,
        visibility: Option<ra::Visibility>,
        element: &impl ra::AstElement,
    ) -> Spanned<ast::VisibilityKind> {
        Spanned {
            span: if let Some(visibility) = visibility.as_ref() {
                self.get_inner_span(visibility)
            } else {
                self.get_inner_span(element).shrink_to_lo()
            },
            node: if let Some(visibility) = visibility {
                match visibility.kind() {
                    ra::VisibilityKind::Pub => ast::VisibilityKind::Public,
                    ra::VisibilityKind::PubCrate => {
                        ast::VisibilityKind::Crate(if visibility.pub_kw().is_some() {
                            ast::CrateSugar::PubCrate
                        } else {
                            ast::CrateSugar::JustCrate
                        })
                    }
                    ra::VisibilityKind::PubSuper => ast::VisibilityKind::Restricted {
                        id: ast::DUMMY_NODE_ID,
                        path: P(self.mk_path_for_ident(
                            self.to_ident_from_token(visibility.super_kw().unwrap()),
                        )),
                    },
                    ra::VisibilityKind::PubSelf => ast::VisibilityKind::Restricted {
                        id: ast::DUMMY_NODE_ID,
                        path: P(self.mk_path_for_ident(
                            self.to_ident_from_token(visibility.self_kw().unwrap()),
                        )),
                    },
                    ra::VisibilityKind::In(path) => ast::VisibilityKind::Restricted {
                        id: ast::DUMMY_NODE_ID,
                        path: P(self.to_simple_path(path, "pub(in path)")),
                    },
                }
            } else {
                ast::VisibilityKind::Inherited
            },
        }
    }

    fn to_variant_data(&self, field_def_list: Option<ra::FieldDefList>) -> ast::VariantData {
        match field_def_list {
            None => ast::VariantData::Unit(ast::DUMMY_NODE_ID),
            Some(ra::FieldDefList::TupleFieldDefList(tuple_field_def_list)) => {
                ast::VariantData::Tuple(
                    tuple_field_def_list
                        .fields()
                        .map(|field| ast::StructField {
                            id: ast::DUMMY_NODE_ID,
                            span: self.get_inner_span(&field),
                            attrs: self.get_attributes(&field).collect(),
                            vis: self.get_visibility(&field),
                            ident: None,
                            ty: self.to_ty_or_err(
                                field.type_ref(),
                                &field,
                                "tuple struct field type",
                            ),
                            is_placeholder: false,
                        })
                        .collect(),
                    ast::DUMMY_NODE_ID,
                )
            }
            Some(ra::FieldDefList::RecordFieldDefList(record_field_def_list)) => {
                ast::VariantData::Struct(
                    record_field_def_list
                        .fields()
                        .map(|field| ast::StructField {
                            id: ast::DUMMY_NODE_ID,
                            span: self.get_inner_span(&field),
                            attrs: self.get_attributes(&field).collect(),
                            vis: self.get_visibility(&field),
                            ident: Some(self.to_ident_from_name_or_err(
                                field.name(),
                                &field,
                                "record field name",
                            )),
                            ty: self.to_ty_or_err(
                                field.ascribed_type(),
                                &field,
                                "record field type",
                            ),
                            is_placeholder: false,
                        })
                        .collect(),
                    false, // TODO: this is a "recovered" flag that is apparently set when the parser encounters errors in the struct; why is it there only for this ast node? should we set it? how?
                )
            }
        }
    }

    fn get_fn_sig(&self, fn_def: &ra::FnDef) -> ast::FnSig {
        let end = if let Some(body) = fn_def.body() {
            inner_start(&body)
        } else if let Some(semi) = fn_def.semi() {
            inner_start(&semi)
        } else {
            start_of_next(fn_def)
        };
        ast::FnSig {
            decl: self.to_fn_decl(
                fn_def.param_list(),
                fn_def.ret_type(),
                end,
                FnParamMode { require_pat: true, require_type: true },
            ),
            header: ast::FnHeader {
                asyncness: self.to_async(fn_def.async_kw()),
                constness: self.to_constness(fn_def.const_kw()),
                ext: self.to_extern(fn_def.abi()),
                unsafety: self.to_unsafe(fn_def.unsafe_kw()),
            },
        }
    }

    fn to_use_tree_or_err(
        &self,
        use_tree_opt: Option<ra::UseTree>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> ast::UseTree {
        self.expect_in(use_tree_opt, element, "use tree", container)
            .map(|use_tree| self.to_use_tree(use_tree))
            .unwrap_or_else(|| {
                let span = self.get_inner_span(element);
                ast::UseTree {
                    span,
                    prefix: self.mk_path_for_ident(ast::Ident::new(kw::Invalid, span)),
                    kind: ast::UseTreeKind::Simple(None, ast::DUMMY_NODE_ID, ast::DUMMY_NODE_ID),
                }
            })
    }

    fn to_use_tree(&self, use_tree: ra::UseTree) -> ast::UseTree {
        let span = self.get_inner_span(&use_tree);
        ast::UseTree {
            span,
            prefix: if let Some(path) = use_tree.path() {
                self.to_simple_path(path, "use tree prefix")
            } else {
                ast::Path { segments: Default::default(), span: span.shrink_to_lo() }
            },
            kind: if use_tree.star().is_some() {
                ast::UseTreeKind::Glob
            } else if let Some(use_tree_list) = use_tree.use_tree_list() {
                ast::UseTreeKind::Nested(
                    use_tree_list
                        .use_trees()
                        .map(|use_tree| (self.to_use_tree(use_tree), ast::DUMMY_NODE_ID))
                        .collect(),
                )
            } else {
                ast::UseTreeKind::Simple(
                    if let Some(alias) = use_tree.alias() {
                        Some(self.to_ident_from_name_or_err(alias.name(), &alias, "use as"))
                    } else {
                        None
                    },
                    ast::DUMMY_NODE_ID,
                    ast::DUMMY_NODE_ID,
                )
            },
        }
    }

    fn to_items<T: TryFrom<ast::ItemKind> + 'static>(
        &self,
        owner: Option<impl ra::ModuleItemOwner>,
        container: &str,
    ) -> Vec<P<ast::Item<T>>> {
        owner
            .into_iter()
            .flat_map(|owner| owner.items().flat_map(|item| self.to_item_opt(item, container)))
            .collect()
    }

    fn to_module_item(
        &self,
        module_item: ra::ModuleItem,
        container: &str,
    ) -> P<ast::Item<ast::ItemKind>> {
        self.to_item_opt(module_item, container).unwrap() /* guaranteed to succeed because it's a try_into from T to T */
    }

    fn to_item_opt<T: TryFrom<ast::ItemKind> + 'static>(
        &self,
        module_item: ra::ModuleItem,
        container: &str,
    ) -> Option<P<ast::Item<T>>> {
        let span = self.get_inner_span(&module_item); // must be inner for extern crate
        let attrs = self.get_attributes(&module_item).collect();
        let tokens = self.get_collected_tokens_from_range(inner_range(&module_item));
        let vis = self.get_visibility(&module_item);

        // TODO: global asm, trait alias, macro 2.0 def
        let (ident, kind) = match module_item {
            ra::ModuleItem::ConstDef(const_def) => (
                self.to_ident_from_name_or_err(const_def.name(), &const_def, "const definition"),
                ast::ItemKind::Const(
                    self.to_defaultness(const_def.default_kw()),
                    self.to_ty_or_err(const_def.ascribed_type(), &const_def, "const definition"),
                    const_def.body().map(|expr| self.to_expr(expr)),
                ),
            ),
            ra::ModuleItem::EnumDef(enum_def) => (
                self.to_ident_from_name_or_err(enum_def.name(), &enum_def, "enum definition"),
                ast::ItemKind::Enum(
                    ast::EnumDef {
                        variants: enum_def
                            .variant_list()
                            .into_iter()
                            .flat_map(|variant_list| {
                                variant_list.variants().map(|variant| ast::Variant {
                                    id: ast::DUMMY_NODE_ID,
                                    span: self.get_inner_span(&variant),
                                    attrs: self.get_attributes(&variant).collect(),
                                    vis: self.to_visibility(variant.visibility(), &variant),
                                    disr_expr: variant.expr().map(|expr| self.to_anon_const(expr)),
                                    data: self.to_variant_data(variant.field_def_list()),
                                    ident: self.to_ident_from_name_or_err(
                                        variant.name(),
                                        &variant,
                                        "enum variant name",
                                    ),
                                    is_placeholder: false,
                                })
                            })
                            .collect(),
                    },
                    self.get_generics_for_named(
                        &enum_def,
                        || enum_def.enum_kw().map(|element| inner_end(&element)),
                        || enum_def.variant_list().map(|element| end_of_prev(&element)),
                    ),
                ),
            ),
            ra::ModuleItem::ExternBlock(extern_block) => (
                ast::Ident::invalid(),
                ast::ItemKind::ForeignMod(ast::ForeignMod {
                    abi: if let Some(abi) =
                        self.expect_in(extern_block.abi(), &extern_block, "extern", "extern block")
                    {
                        if let Some(str) = abi.string() {
                            Some(self.to_str_lit(str))
                        } else {
                            None
                        }
                    } else {
                        None
                    },
                    items: self.to_items(extern_block.extern_item_list(), "extern block"),
                }),
            ),
            ra::ModuleItem::ExternCrateItem(extern_crate_item) => {
                if let Some(alias) = extern_crate_item.alias() {
                    (
                        self.to_ident_from_name_or_err(alias.name(), &alias, "extern crate alias"),
                        ast::ItemKind::ExternCrate(Some(
                            self.to_ident_from_name_ref_or_err(
                                extern_crate_item.name_ref(),
                                &extern_crate_item,
                                "extern crate reference",
                            )
                            .name,
                        )),
                    )
                } else {
                    (
                        self.to_ident_from_name_ref_or_err(
                            extern_crate_item.name_ref(),
                            &extern_crate_item,
                            "extern crate",
                        ),
                        ast::ItemKind::ExternCrate(None),
                    )
                }
            }
            ra::ModuleItem::FnDef(fn_def) => (
                self.to_ident_from_name_or_err(fn_def.name(), &fn_def, "function definition"),
                ast::ItemKind::Fn(
                    self.to_defaultness(fn_def.default_kw()),
                    self.get_fn_sig(&fn_def),
                    self.get_generics_for_named(
                        &fn_def,
                        || fn_def.fn_kw().map(|element| inner_end(&element)),
                        || {
                            fn_def
                                .body()
                                .map(|element| end_of_prev(&element))
                                .or(fn_def.semi().map(|element| end_of_prev(&element)))
                        },
                    ),
                    fn_def
                        .body()
                        .and_then(|block_expr| block_expr.block())
                        .map(|block| self.to_block(block)),
                ),
            ),
            ra::ModuleItem::ImplDef(impl_def) => (
                ast::Ident::invalid(),
                ast::ItemKind::Impl {
                    constness: self.to_constness(impl_def.const_kw()),
                    defaultness: self.to_defaultness(impl_def.default_kw()),
                    unsafety: self.to_unsafe(impl_def.unsafe_kw()),
                    polarity: match impl_def.excl() {
                        Some(excl) => ast::ImplPolarity::Negative(self.get_inner_span(&excl)),
                        None => ast::ImplPolarity::Positive,
                    },
                    generics: self.get_generics_for_unnamed(
                        &impl_def,
                        || impl_def.impl_kw().map(|element| inner_end(&element)),
                        || impl_def.item_list().map(|element| end_of_prev(&element)),
                    ),
                    self_ty: self.to_ty_or_err(
                        impl_def.target_type(),
                        &impl_def,
                        "impl target type",
                    ),
                    of_trait: impl_def.target_trait().map(|trait_type_ref| ast::TraitRef {
                        ref_id: ast::DUMMY_NODE_ID,
                        path: self.to_trait_path_from_type(trait_type_ref),
                    }),
                    items: self.to_items(impl_def.item_list(), "impl"),
                },
            ),
            ra::ModuleItem::MacroCall(macro_call) => {
                if let Some(item_kind) = self.get_item_kind_opt_from_macro_call(&macro_call) {
                    (
                        self.to_ident_from_name_or_err(
                            macro_call.name(),
                            &macro_call,
                            "macro_rules! macro definition",
                        ),
                        item_kind,
                    )
                } else {
                    (ast::Ident::invalid(), ast::ItemKind::MacCall(self.to_mac_call(macro_call)))
                }
            }
            ra::ModuleItem::Module(module) => (
                self.to_ident_from_name_or_err(module.name(), &module, "module definition"),
                ast::ItemKind::Mod(if let Some(item_list) = module.item_list() {
                    ast::Mod {
                        inner: self.to_span(TextRange::from_to(
                            item_list
                                .l_curly()
                                .map(|l_curly| start_of_next(&l_curly))
                                .unwrap_or_else(|| inner_start(&item_list)),
                            item_list
                                .r_curly()
                                .map(|r_curly| inner_end(&r_curly))
                                .unwrap_or_else(|| inner_end(&item_list)),
                        )),
                        items: self.to_items(Some(item_list), "module"),
                        inline: true,
                    }
                } else {
                    ast::Mod { inner: Span::default(), items: Default::default(), inline: false }
                }),
            ),
            ra::ModuleItem::StaticDef(static_def) => (
                self.to_ident_from_name_or_err(
                    static_def.name(),
                    &static_def,
                    "static variable definition",
                ),
                ast::ItemKind::Static(
                    self.to_ty_or_err(
                        static_def.ascribed_type(),
                        &static_def,
                        "static variable definition",
                    ),
                    self.to_mutability(static_def.mut_kw()),
                    static_def.body().map(|expr| self.to_expr(expr)),
                ),
            ),
            ra::ModuleItem::StructDef(struct_def) => {
                (
                self.to_ident_from_name_or_err(struct_def.name(), &struct_def, "struct definition"),
                ast::ItemKind::Struct(
                    self.to_variant_data(struct_def.field_def_list()),
                    self.get_generics_for_named(&struct_def,
                        || struct_def.struct_kw().map(|element| inner_end(&element)),
                        || None
                            .or(
                                struct_def.field_def_list().and_then(|field_def_list| // where clauses are allowed after tuple struct fields
                                    match field_def_list {ra::FieldDefList::RecordFieldDefList(record_field_def_list) => Some(record_field_def_list), _ => None}
                                ).map(|element| end_of_prev(&element))
                            ).or(struct_def.semi().map(|element| end_of_prev(&element)))
                    )
                )
            )
            }
            ra::ModuleItem::TraitDef(trait_def) => (
                self.to_ident_from_name_or_err(trait_def.name(), &trait_def, "trait definition"),
                ast::ItemKind::Trait(
                    if trait_def.auto_kw().is_some() { ast::IsAuto::Yes } else { ast::IsAuto::No },
                    self.to_unsafe(trait_def.unsafe_kw()),
                    self.get_generics_for_named(
                        &trait_def,
                        || trait_def.trait_kw().map(|element| inner_end(&element)),
                        || trait_def.item_list().map(|element| end_of_prev(&element)),
                    ),
                    self.to_generic_bounds(trait_def.type_bound_list()),
                    self.to_items(trait_def.item_list(), "trait definition"),
                ),
            ),
            ra::ModuleItem::TypeAliasDef(type_alias_def) => (
                self.to_ident_from_name_or_err(
                    type_alias_def.name(),
                    &type_alias_def,
                    "type alias definition",
                ),
                ast::ItemKind::TyAlias(
                    self.to_defaultness(type_alias_def.default_kw()),
                    self.get_generics_for_named(
                        &type_alias_def,
                        || type_alias_def.type_kw().map(|element| inner_end(&element)),
                        || {
                            type_alias_def
                                .eq()
                                .map(|element| end_of_prev(&element))
                                .or(type_alias_def.semi().map(|element| end_of_prev(&element)))
                        },
                    ),
                    self.to_generic_bounds(type_alias_def.type_bound_list()),
                    type_alias_def.type_ref().map(|type_ref| self.to_ty(type_ref)),
                ),
            ),
            ra::ModuleItem::UnionDef(union_def) => (
                self.to_ident_from_name_or_err(union_def.name(), &union_def, "union definition"),
                ast::ItemKind::Union(
                    self.to_variant_data(
                        if let Some(record_field_def_list) = self.expect_in(
                            union_def.record_field_def_list(),
                            &union_def,
                            "field list",
                            "union",
                        ) {
                            Some(ra::FieldDefList::RecordFieldDefList(record_field_def_list))
                        } else {
                            None
                        },
                    ),
                    self.get_generics_for_named(
                        &union_def,
                        || union_def.union_kw().map(|element| inner_end(&element)),
                        || union_def.record_field_def_list().map(|element| end_of_prev(&element)),
                    ),
                ),
            ),
            ra::ModuleItem::UseItem(use_item) => (
                ast::Ident::invalid(),
                ast::ItemKind::Use(P(self.to_use_tree_or_err(
                    use_item.use_tree(),
                    &use_item,
                    "use",
                ))),
            ),
        };
        if let Ok(kind) = kind.try_into() {
            Some(P(ast::Item {
                attrs,
                id: ast::DUMMY_NODE_ID, // this is correct
                span,
                vis,
                ident,
                kind,
                tokens,
            }))
        } else {
            self.error_at_span(span, &format!("unexpected item kind in {}", container));
            None
        }
    }

    fn get_attributes<'a>(
        &'a self,
        node: &'a impl ra::AttrsOwner,
    ) -> impl Iterator<Item = ast::Attribute> + 'a {
        self.to_attributes(node.attr_or_comments())
    }

    fn to_mac_args_from_token_tree(&self, token_tree: ra::TokenTree) -> ast::MacArgs {
        let left = self.expect_in(
            token_tree.left_delimiter(),
            &token_tree,
            "left delimiter among '{', '(', '['",
            "attribute value",
        );
        let right = self.expect_in(
            token_tree.right_delimiter(),
            &token_tree,
            "right delimiter among '}', ')', ']'",
            "attribute value",
        );

        let ast_left = left.as_ref().map(|left| match left {
            ra::LeftDelimiter::LParen(_) => ast::MacDelimiter::Parenthesis,
            ra::LeftDelimiter::LBrack(_) => ast::MacDelimiter::Bracket,
            ra::LeftDelimiter::LCurly(_) => ast::MacDelimiter::Brace,
        });

        let ast_right = right.as_ref().map(|right| match right {
            ra::RightDelimiter::RParen(_) => ast::MacDelimiter::Parenthesis,
            ra::RightDelimiter::RBrack(_) => ast::MacDelimiter::Bracket,
            ra::RightDelimiter::RCurly(_) => ast::MacDelimiter::Brace,
        });

        let ast_delim = match (ast_left, ast_right) {
            (Some(ast_left), Some(ast_right)) if ast_left == ast_right => Some(ast_left),
            (Some(_), Some(_)) => {
                self.missing_in(
                    &token_tree,
                    "right delimiter matching with left delimiter",
                    "attribute value",
                );
                None
            }
            _ => None,
        }
        .unwrap_or(ast::MacDelimiter::Brace);

        let open_range = left
            .map(|left| inner_range(&left))
            .unwrap_or_else(|| empty_text_range(inner_start(&token_tree)));
        let close_range = right
            .map(|right| inner_range(&right))
            .unwrap_or_else(|| empty_text_range(inner_end(&token_tree)));
        let delim_span = rustc_ast::tokenstream::DelimSpan {
            open: self.to_span(open_range),
            close: self.to_span(close_range),
        };
        ast::MacArgs::Delimited(
            delim_span,
            ast_delim,
            self.get_tokens_from_range(TextRange::from_to(open_range.end(), close_range.start()))
                .unwrap_or_else(|| TokenStream::default()),
        )
    }

    fn to_attributes<'a>(
        &'a self,
        attr_or_comments: ra::AstChildElements<ra::AttrOrComment>,
    ) -> impl Iterator<Item = ast::Attribute> + 'a {
        attr_or_comments.filter_map(move |attr_or_comment: ra::AttrOrComment| match attr_or_comment
        {
            ra::AttrOrComment::Attr(attr) => {
                let mac_args = match attr.input() {
                    Some(ra::AttrInput::TokenTree(token_tree)) => {
                        self.to_mac_args_from_token_tree(token_tree)
                    }
                    Some(ra::AttrInput::Literal(literal)) => ast::MacArgs::Eq(
                        self.expect_in(attr.eq(), &attr, "equals sign", "literal attribute")
                            .map(|eq| self.get_inner_span(&eq))
                            .unwrap_or_else(|| self.get_inner_span(&literal).shrink_to_lo()),
                        self.to_tokens(literal).unwrap_or_else(|| TokenStream::default()),
                    ),
                    None => ast::MacArgs::Empty,
                };

                Some(rustc_ast::attr::mk_attr(
                    match attr.kind() {
                        ra::AttrKind::Inner => ast::AttrStyle::Inner,
                        ra::AttrKind::Outer => ast::AttrStyle::Outer,
                    },
                    self.to_simple_path_or_err(attr.path(), &attr, "attribute path"),
                    mac_args,
                    self.get_inner_span(&attr),
                ))
            }
            ra::AttrOrComment::Comment(comment) => comment.kind().doc.map(|doc| {
                let span = self.get_inner_span(&comment);
                rustc_ast::attr::mk_doc_comment(
                    match doc {
                        ra::CommentPlacement::Outer => ast::AttrStyle::Outer,
                        ra::CommentPlacement::Inner => ast::AttrStyle::Inner,
                    },
                    self.to_symbol_from_token(comment),
                    span,
                )
            }),
        })
    }

    fn invalid_qself(&self, span: Span, container: &str) {
        self.error_at_span(
            span,
            &format!("{} must not include a fully qualified path prefix", container),
        );
    }

    fn to_simple_path(&self, path: ra::Path, container: &str) -> ast::Path {
        let (qself, path) = self.to_qself_and_path(path);
        if let Some((_qself, qself_span)) = qself {
            self.invalid_qself(qself_span, container);
        }
        path
    }

    fn to_simple_path_or_err(
        &self,
        path: Option<ra::Path>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> ast::Path {
        let (qself, path) = self.to_qself_and_path_or_err(path, element, container);
        if let Some((_qself, qself_span)) = qself {
            self.invalid_qself(qself_span, container);
        }
        path
    }

    fn to_trait_path_from_path_type(&self, node: ra::PathType) -> ast::Path {
        self.to_simple_path_or_err(node.path(), &node, "trait reference")
    }

    fn to_trait_path_from_type(&self, node: ra::TypeRef) -> ast::Path {
        match node {
            ra::TypeRef::PathType(path_type) => self.to_trait_path_from_path_type(path_type),
            _ => {
                self.error(&node, "non-path type used as trait reference");
                self.mk_path_for_ident(ast::Ident::new(kw::Invalid, self.get_inner_span(&node)))
            }
        }
    }

    fn to_qself_and_path_or_err(
        &self,
        path: Option<ra::Path>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> (Option<(ast::QSelf, Span)>, ast::Path) {
        self.expect_in(path, element, "path", container)
            .map(|path| self.to_qself_and_path(path))
            .unwrap_or_else(|| {
                (
                    None,
                    ast::Path { span: self.get_inner_span(element), segments: Default::default() },
                )
            })
    }

    fn to_qself_and_path(&self, node: ra::Path) -> (Option<(ast::QSelf, Span)>, ast::Path) {
        let mut qself = None;
        let path = ast::Path {
            span: self.get_inner_span(&node),
            segments: {
                let mut segments = Vec::new();
                let mut path_opt = Some(node);
                while let Some(path) = path_opt {
                    let qualifier = path.qualifier();
                    let ast_segment = self.expect_in(path.segment(), &path, "path segment", "path").map(|segment| {
                        let mut ast_segment = self.expect_in(segment.kind(), &segment, "valid path segment", "path segment").map(|kind|
                            match kind {
                                ra::PathSegmentKind::Name(name_ref) => Some(ast::PathSegment::from_ident(self.to_ident_from_name_ref(name_ref))),
                                ra::PathSegmentKind::CrateKw => Some(ast::PathSegment::from_ident(ast::Ident::new(kw::Crate, self.get_inner_span(&segment)))),
                                ra::PathSegmentKind::SuperKw => Some(ast::PathSegment::from_ident(ast::Ident::new(kw::Super, self.get_inner_span(&segment)))),
                                ra::PathSegmentKind::SelfKw => Some(ast::PathSegment::from_ident(ast::Ident::new(kw::SelfLower, self.get_inner_span(&segment)))),
                                ra::PathSegmentKind::Type {type_ref, trait_ref} => {
                                    if qualifier.is_some() {
                                        self.error(&segment, "a qualified path segment can only be the first segment in a path");
                                    } else {
                                        let mut path_span = if let Some(type_ref) = type_ref.as_ref() {
                                            self.get_inner_span(type_ref).shrink_to_hi()
                                        } else {
                                            // TODO: is this correct? it should be
                                            if let Some(l_angle) = segment.l_angle() {
                                                self.get_inner_span(&l_angle).shrink_to_lo()
                                            } else {
                                                self.get_inner_span(&segment).shrink_to_lo()
                                            }
                                        };
                                        let mut position = 0;
                                        if let Some(trait_ref) = trait_ref {
                                            let trait_path = self.to_trait_path_from_path_type(trait_ref);
                                            for trait_segment in trait_path.segments.into_iter().rev() {
                                                segments.push(trait_segment);
                                                position += 1;
                                            }
                                            path_span = trait_path.span;
                                        }
                                        qself = Some((ast::QSelf {
                                            path_span,
                                            ty: self.to_ty_or_err(type_ref, &segment, "fully qualified path segment"),
                                            position
                                        }, self.get_inner_span(&segment)));
                                    }
                                    None
                                }
                            }
                        ).unwrap_or_else(|| Some(ast::PathSegment {
                            id: ast::DUMMY_NODE_ID,
                            ident: ast::Ident::new(kw::Invalid, self.get_inner_span(&segment)),
                            args: None
                        }));

                        if let Some(ast_segment) = ast_segment.as_mut() {
                            if let Some(type_arg_list) = segment.type_arg_list() {
                                ast_segment.args = Some(P(ast::GenericArgs::AngleBracketed(self.to_angle_bracketed_args(&type_arg_list))))
                            } else if let Some(param_list) = segment.param_list() {
                                ast_segment.args = Some(P(ast::GenericArgs::Parenthesized(ast::ParenthesizedArgs {
                                    span: self.to_span(TextRange::from_to(inner_start(&segment), inner_end(&param_list))),
                                    inputs: self.to_tys(param_list),
                                    output: self.to_fn_ret_ty(segment.ret_type(), start_of_next(&segment)),
                                })));
                            }
                        }
                        ast_segment
                    }).unwrap_or_else(|| Some(ast::PathSegment {
                        id: ast::DUMMY_NODE_ID,
                        ident: ast::Ident::new(kw::Invalid, self.get_inner_span(&path)),
                        args: None
                    }));

                    if let Some(ast_segment) = ast_segment {
                        segments.push(ast_segment);
                    }
                    path_opt = qualifier;
                    if path_opt.is_none() {
                        match path.segment().and_then(|x| x.coloncolon()) {
                            Some(coloncolon) => segments.push(ast::PathSegment::path_root(
                                self.get_inner_span(&coloncolon),
                            )),
                            _ => {}
                        }
                    }
                }
                segments.reverse();
                segments
            },
        };
        (qself, path)
    }

    fn to_fn_ret_ty(&self, ret_type: Option<ra::RetType>, end: TextUnit) -> ast::FnRetTy {
        if let Some(ret_type) = ret_type {
            ast::FnRetTy::Ty(self.to_ty_or_err(ret_type.type_ref(), &ret_type, "return type"))
        } else {
            ast::FnRetTy::Default(self.to_span_from_unit(end))
        }
    }

    fn to_tys(&self, param_list: ra::ParamList) -> Vec<P<ast::Ty>> {
        param_list
            .params()
            .map(|param| self.to_ty_or_err(param.ascribed_type(), &param, "parameter"))
            .collect()
    }

    fn to_ty_or_err(
        &self,
        type_ref_opt: Option<ra::TypeRef>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> P<ast::Ty> {
        self.expect_in(type_ref_opt, element, "type", container)
            .map(|type_ref| self.to_ty(type_ref))
            .unwrap_or_else(|| {
                P(ast::Ty {
                    id: ast::DUMMY_NODE_ID,
                    span: self.get_inner_span(element),
                    kind: ast::TyKind::Err,
                })
            })
    }

    fn get_token(&self, token: &impl ra::AstToken) -> token::Token {
        let mut srdr = lexer::StringReader::retokenize(&self.sess, self.get_inner_span(token));
        srdr.next_token()
    }

    fn to_lit_or_err(
        &self,
        token: Option<ra::LiteralToken>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> ast::Lit {
        self.expect_in(token, element, "literal", container)
            .map(|token| self.to_lit(token))
            .unwrap_or_else(|| ast::Lit {
                span: self.get_inner_span(element),
                kind: ast::LitKind::Err(kw::Invalid),
                token: token::Lit { kind: token::LitKind::Err, suffix: None, symbol: kw::Invalid },
            })
    }

    fn to_lit(&self, token: ra::LiteralToken) -> ast::Lit {
        let span = self.get_inner_span(&token);
        ast::Lit::from_token(&self.get_token(&token)).unwrap_or_else(|_| {
            self.error_at_span(span, "invalid literal token");
            ast::Lit {
                span,
                kind: ast::LitKind::Err(kw::Invalid),
                token: token::Lit { kind: token::LitKind::Err, suffix: None, symbol: kw::Invalid },
            }
        })
    }

    fn to_str_lit(&self, string: ra::String) -> ast::StrLit {
        let lit = self.to_lit(ra::LiteralToken::String(string));
        match lit.kind {
            ast::LitKind::Str(symbol_unescaped, style) => ast::StrLit {
                span: lit.span,
                style,
                symbol: lit.token.symbol,
                suffix: lit.token.suffix,
                symbol_unescaped,
            },
            _ => ast::StrLit {
                span: lit.span,
                style: ast::StrStyle::Cooked,
                symbol: lit.token.symbol,
                suffix: lit.token.suffix,
                symbol_unescaped: lit.token.symbol,
            },
        }
    }

    fn to_async(&self, async_kw: Option<ra::AsyncKw>) -> ast::Async {
        if let Some(async_kw) = async_kw {
            ast::Async::Yes {
                span: self.get_inner_span(&async_kw),
                closure_id: ast::DUMMY_NODE_ID,
                return_impl_trait_id: ast::DUMMY_NODE_ID,
            }
        } else {
            ast::Async::No
        }
    }

    fn to_extern(&self, abi: Option<ra::Abi>) -> ast::Extern {
        if let Some(abi) = abi {
            if let Some(str) = abi.string() {
                ast::Extern::Explicit(self.to_str_lit(str))
            } else {
                ast::Extern::Implicit
            }
        } else {
            ast::Extern::None
        }
    }

    fn to_unsafe(&self, unsafe_kw: Option<ra::UnsafeKw>) -> ast::Unsafe {
        if let Some(unsafe_kw) = unsafe_kw {
            ast::Unsafe::Yes(self.get_inner_span(&unsafe_kw))
        } else {
            ast::Unsafe::No
        }
    }

    fn to_bare_fn_ty(
        &self,
        fn_pointer_type: ra::FnPointerType,
        type_param_list_opt: Option<ra::TypeParamList>,
    ) -> P<ast::BareFnTy> {
        P(ast::BareFnTy {
            generic_params: self.to_generic_params(type_param_list_opt),
            unsafety: self.to_unsafe(fn_pointer_type.unsafe_kw()),
            ext: self.to_extern(fn_pointer_type.abi()),
            decl: self.to_fn_decl(
                fn_pointer_type.param_list(),
                fn_pointer_type.ret_type(),
                start_of_next(&fn_pointer_type),
                FnParamMode { require_pat: false, require_type: true },
            ),
        })
    }

    fn to_fn_decl(
        &self,
        param_list: Option<ra::ParamList>,
        ret_type: Option<ra::RetType>,
        end: TextUnit,
        fn_param_mode: FnParamMode,
    ) -> P<ast::FnDecl> {
        P(ast::FnDecl {
            inputs: self.to_params(param_list, fn_param_mode),
            output: self.to_fn_ret_ty(ret_type, end),
        })
    }

    fn to_pat_or_err(
        &self,
        pat_opt: Option<ra::Pat>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> P<ast::Pat> {
        self.expect_in(pat_opt, element, "pattern", container)
            .map(|pat| self.to_pat(pat))
            .unwrap_or_else(|| {
                let span = self.get_inner_span(element);
                P(ast::Pat {
                    id: ast::DUMMY_NODE_ID,
                    span,
                    kind: ast::PatKind::Ident(
                        ast::BindingMode::ByValue(ast::Mutability::Not),
                        ast::Ident::new(kw::Invalid, span),
                        None,
                    ),
                })
            })
    }

    fn to_mutability(&self, mut_kw: Option<ra::MutKw>) -> ast::Mutability {
        if mut_kw.is_some() {
            ast::Mutability::Mut
        } else {
            ast::Mutability::Not
        }
    }

    fn to_mutability_from_const(&self, const_kw: Option<ra::ConstKw>) -> ast::Mutability {
        if const_kw.is_none() {
            ast::Mutability::Mut
        } else {
            ast::Mutability::Not
        }
    }

    fn to_borrow_kind(&self, raw_kw: Option<ra::RawKw>) -> ast::BorrowKind {
        if raw_kw.is_some() {
            ast::BorrowKind::Raw
        } else {
            ast::BorrowKind::Ref
        }
    }

    fn to_binding_mode(
        &self,
        ref_kw: Option<ra::RefKw>,
        mut_kw: Option<ra::MutKw>,
    ) -> ast::BindingMode {
        let mutability = self.to_mutability(mut_kw);
        if ref_kw.is_some() {
            ast::BindingMode::ByRef(mutability)
        } else {
            ast::BindingMode::ByValue(mutability)
        }
    }

    fn to_range_end(&self, separator: ra::RangeSeparator) -> ast::RangeEnd {
        match separator {
            ra::RangeSeparator::Dotdot(_) => ast::RangeEnd::Excluded,
            ra::RangeSeparator::Dotdotdot(_) => {
                ast::RangeEnd::Included(ast::RangeSyntax::DotDotDot)
            }
            ra::RangeSeparator::Dotdoteq(_) => ast::RangeEnd::Included(ast::RangeSyntax::DotDotEq),
        }
    }

    fn to_range_end_or_err(
        &self,
        separator: Option<ra::RangeSeparator>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> ast::RangeEnd {
        if let Some(separator) = self.expect_in(separator, element, "range separator", container) {
            self.to_range_end(separator)
        } else {
            ast::RangeEnd::Excluded
        }
    }

    // currently only for usage to parse RangeExpr
    fn to_expr_from_pat(&self, pat: ra::Pat) -> P<ast::Expr> {
        match pat {
            ra::Pat::LiteralPat(literal_pat) => self.to_expr_from_literal_pat(literal_pat),
            ra::Pat::BindPat(bind_pat) => {
                self.invalid_token(bind_pat.ref_kw(), "expression inside pattern");
                self.invalid_token(bind_pat.mut_kw(), "expression inside pattern");
                self.invalid_element(
                    bind_pat.pat(),
                    &bind_pat,
                    "guarded pattern",
                    "expression inside pattern",
                );
                self.mk_expr_for_ident(self.to_ident_from_name_or_err(
                    bind_pat.name(),
                    &bind_pat,
                    "expression inside pattern",
                ))
            }
            ra::Pat::PathPat(path_pat) => {
                let (qself, path) =
                    self.to_qself_and_path_or_err(path_pat.path(), &path_pat, "path pattern");
                P(ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    attrs: Default::default(),
                    span: self.get_inner_span(&path_pat),
                    kind: ast::ExprKind::Path(qself.map(|(qself, _)| qself), path),
                })
            }
            node => {
                self.error(&node, "invalid expression inside pattern");
                P(ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    span: self.get_inner_span(&node),
                    attrs: Default::default(),
                    kind: ast::ExprKind::Err,
                })
            }
        }
    }

    fn to_expr_from_condition_or_err(
        &self,
        condition: Option<ra::Condition>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> P<ast::Expr> {
        if let Some(condition) = self.expect_in(condition, element, "condition", container) {
            let expr = self.to_expr_or_err(condition.expr(), &condition, "condition expression");
            if let Some(pat) = condition.pat() {
                P(ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    span: self.get_inner_span(&condition),
                    attrs: Default::default(),
                    kind: ast::ExprKind::Let(self.to_pat(pat), expr),
                })
            } else {
                expr
            }
        } else {
            P(ast::Expr {
                id: ast::DUMMY_NODE_ID,
                span: self.get_inner_span(element),
                attrs: Default::default(),
                kind: ast::ExprKind::Err,
            })
        }
    }

    fn to_expr_from_literal_pat(&self, literal_pat: ra::LiteralPat) -> P<ast::Expr> {
        P(ast::Expr {
            id: ast::DUMMY_NODE_ID,
            span: self.get_inner_span(&literal_pat),
            attrs: Default::default(),
            kind: ast::ExprKind::Lit(self.to_lit_or_err(
                literal_pat.literal().and_then(|literal| literal.literal_token()),
                &literal_pat,
                "literal pattern",
            )),
        })
    }

    fn to_pat(&self, pat: ra::Pat) -> P<ast::Pat> {
        let span = self.get_inner_span(&pat);
        let kind = match pat {
            ra::Pat::BindPat(bind_pat) => ast::PatKind::Ident(
                self.to_binding_mode(bind_pat.ref_kw(), bind_pat.mut_kw()),
                self.to_ident_from_name_or_err(bind_pat.name(), &bind_pat, "bind pattern"),
                bind_pat.pat().map(|inner_pat| self.to_pat(inner_pat)),
            ),
            ra::Pat::LiteralPat(literal_pat) => {
                ast::PatKind::Lit(self.to_expr_from_literal_pat(literal_pat))
            }
            ra::Pat::OrPat(or_pat) => {
                ast::PatKind::Or(or_pat.pats().map(|inner_pat| self.to_pat(inner_pat)).collect())
            }
            ra::Pat::PathPat(path_pat) => {
                let (qself, path) =
                    self.to_qself_and_path_or_err(path_pat.path(), &path_pat, "path pattern");
                ast::PatKind::Path(qself.map(|(qself, _)| qself), path)
            }
            ra::Pat::RefPat(ref_pat) => ast::PatKind::Ref(
                self.to_pat_or_err(ref_pat.pat(), &ref_pat, "reference pattern"),
                self.to_mutability(ref_pat.mut_kw()),
            ),
            ra::Pat::TuplePat(tuple_pat) => ast::PatKind::Tuple(
                tuple_pat.args().map(|inner_pat| self.to_pat(inner_pat)).collect(),
            ),
            ra::Pat::TupleStructPat(tuple_struct_pat) => ast::PatKind::TupleStruct(
                self.to_simple_path_or_err(
                    tuple_struct_pat.path(),
                    &tuple_struct_pat,
                    "tuple struct pattern",
                ),
                tuple_struct_pat.args().map(|arg| self.to_pat(arg)).collect(),
            ),
            ra::Pat::PlaceholderPat(_placeholder_pat) => ast::PatKind::Wild,
            ra::Pat::ParenPat(paren_pat) => ast::PatKind::Paren(self.to_pat_or_err(
                paren_pat.pat(),
                &paren_pat,
                "parenthesized pattern",
            )),
            ra::Pat::BoxPat(box_pat) => {
                ast::PatKind::Box(self.to_pat_or_err(box_pat.pat(), &box_pat, "box pattern"))
            }
            ra::Pat::DotDotPat(_) => ast::PatKind::Rest,
            ra::Pat::SlicePat(slice_pat) => {
                ast::PatKind::Slice(slice_pat.args().map(|arg| self.to_pat(arg)).collect())
            }
            ra::Pat::RangePat(range_pat) => {
                let separator = range_pat.range_separator();
                ast::PatKind::Range(
                    range_pat.start().map(|inner_pat| self.to_expr_from_pat(inner_pat)),
                    range_pat.end().map(|inner_pat| self.to_expr_from_pat(inner_pat)),
                    Spanned {
                        span: if let Some(separator) = separator.as_ref() {
                            self.get_inner_span(separator)
                        } else {
                            self.get_inner_span(&range_pat)
                        },
                        node: self.to_range_end_or_err(separator, &range_pat, "range pattern"),
                    },
                )
            }
            ra::Pat::RecordPat(record_pat) => {
                let (fields, dotdot) =
                    if let Some(record_field_pat_list) = record_pat.record_field_pat_list() {
                        (
                            record_field_pat_list
                                .pats()
                                .map(|inner_pat| match inner_pat {
                                    ra::RecordInnerPat::RecordFieldPat(record_field_pat) => {
                                        ast::FieldPat {
                                            id: ast::DUMMY_NODE_ID,
                                            span: self.get_inner_span(&record_field_pat),
                                            attrs: self
                                                .get_attributes(&record_field_pat)
                                                .collect::<Vec<_>>()
                                                .into(),
                                            pat: self.to_pat_or_err(
                                                record_field_pat.pat(),
                                                &record_field_pat,
                                                "record field pattern",
                                            ),
                                            ident: self.to_ident_from_name_or_err(
                                                record_field_pat.name(),
                                                &record_field_pat,
                                                "record field pattern",
                                            ),
                                            is_shorthand: false,
                                            is_placeholder: false,
                                        }
                                    }
                                    ra::RecordInnerPat::BindPat(bind_pat) => ast::FieldPat {
                                        id: ast::DUMMY_NODE_ID,
                                        span: self.get_inner_span(&bind_pat),
                                        attrs: self
                                            .get_attributes(&bind_pat)
                                            .collect::<Vec<_>>()
                                            .into(),
                                        ident: self.to_ident_from_name_or_err(
                                            bind_pat.name(),
                                            &bind_pat,
                                            "bind pattern",
                                        ),
                                        pat: self.to_pat(ra::Pat::BindPat(bind_pat)),
                                        is_shorthand: true,
                                        is_placeholder: false,
                                    },
                                })
                                .collect(),
                            record_field_pat_list.dotdot().is_some(),
                        )
                    } else {
                        (Default::default(), false)
                    };

                ast::PatKind::Struct(
                    self.to_simple_path_or_err(record_pat.path(), &record_pat, "record pattern"),
                    fields,
                    dotdot,
                )
            }
            ra::Pat::MacroCall(macro_call) => ast::PatKind::MacCall(self.to_mac_call(macro_call)),
        };
        P(ast::Pat { id: ast::DUMMY_NODE_ID, span, kind })
    }

    fn to_param_from_self_param(&self, self_param: ra::SelfParam) -> ast::Param {
        let self_ident = if let Some(self_kw) =
            self.expect_in(self_param.self_kw(), &self_param, "self keyword", "self parameter")
        {
            self.to_ident_from_token(self_kw)
        } else {
            ast::Ident::new(kw::Invalid, self.get_inner_span(&self_param))
        };

        let span = self.get_inner_span(&self_param);
        ast::Param {
            id: ast::DUMMY_NODE_ID,
            span,
            attrs: self.get_attributes(&self_param).collect::<Vec<_>>().into(),
            ty: if let Some(self_type) = self_param.ascribed_type() {
                self.to_ty(self_type)
            } else {
                let implicit_self_ty =
                    P(ast::Ty { id: ast::DUMMY_NODE_ID, span, kind: ast::TyKind::ImplicitSelf });
                if let Some(_) = self_param.amp() {
                    P(ast::Ty {
                        id: ast::DUMMY_NODE_ID,
                        span: self.get_inner_span(&self_param),
                        kind: ast::TyKind::Rptr(
                            self_param.lifetime().map(|t| self.to_lifetime(t)),
                            ast::MutTy {
                                mutbl: self.to_mutability(self_param.amp_mut_kw()),
                                ty: implicit_self_ty,
                            },
                        ),
                    })
                } else {
                    implicit_self_ty
                }
            },
            pat: P(ast::Pat {
                id: ast::DUMMY_NODE_ID,
                span: self.get_inner_span(&self_param),
                kind: ast::PatKind::Ident(
                    ast::BindingMode::ByValue(self.to_mutability(self_param.mut_kw())),
                    self_ident,
                    None,
                ),
            }),
            is_placeholder: false,
        }
    }

    fn to_param(&self, param: ra::Param, fn_param_mode: FnParamMode) -> ast::Param {
        if let Some(dotdotdot) = param.dotdotdot() {
            let span = self.get_inner_span(&dotdotdot);
            ast::Param {
                id: ast::DUMMY_NODE_ID,
                span: self.get_inner_to_next_span(&param),
                attrs: self.get_attributes(&param).collect::<Vec<_>>().into(),
                ty: P(ast::Ty { id: ast::DUMMY_NODE_ID, span, kind: ast::TyKind::CVarArgs }),
                pat: P(ast::Pat {
                    id: ast::DUMMY_NODE_ID,
                    span,
                    kind: ast::PatKind::Ident(
                        ast::BindingMode::ByValue(ast::Mutability::Not),
                        ast::Ident::from_str_and_span("", span),
                        None,
                    ),
                }),
                is_placeholder: false,
            }
        } else {
            ast::Param {
                id: ast::DUMMY_NODE_ID,
                span: self.get_inner_to_next_span(&param),
                attrs: self.get_attributes(&param).collect::<Vec<_>>().into(),
                ty: if fn_param_mode.require_type {
                    self.to_ty_or_err(param.ascribed_type(), &param, "parameter")
                } else {
                    param.ascribed_type().map(|type_ref| self.to_ty(type_ref)).unwrap_or_else(
                        || {
                            P(ast::Ty {
                                id: ast::DUMMY_NODE_ID,
                                span: self.get_last_token_span(&param),
                                kind: ast::TyKind::Infer,
                            })
                        },
                    )
                },
                pat: if fn_param_mode.require_pat {
                    self.to_pat_or_err(param.pat(), &param, "parameter")
                } else {
                    let span = self.get_inner_span(&param);
                    param.pat().map(|pat| self.to_pat(pat)).unwrap_or_else(|| {
                        P(ast::Pat {
                            id: ast::DUMMY_NODE_ID,
                            span,
                            kind: ast::PatKind::Ident(
                                ast::BindingMode::ByValue(ast::Mutability::Not),
                                ast::Ident::from_str_and_span("", span),
                                None,
                            ),
                        })
                    })
                },
                is_placeholder: false,
            }
        }
    }

    fn to_params(
        &self,
        param_list: Option<ra::ParamList>,
        fn_param_mode: FnParamMode,
    ) -> Vec<ast::Param> {
        if let Some(param_list) = param_list {
            if let Some(self_param) = param_list.self_param() {
                Some(self.to_param_from_self_param(self_param))
            } else {
                None
            }
            .into_iter()
            .chain(param_list.params().map(|param| self.to_param(param, fn_param_mode)))
            .collect()
        } else {
            Default::default()
        }
    }

    fn to_ty(&self, mut type_ref: ra::TypeRef) -> P<ast::Ty> {
        loop {
            let span = self.get_inner_span(&type_ref);
            let kind = match type_ref {
                ra::TypeRef::PathType(path_type) => {
                    let (qself, path) =
                        self.to_qself_and_path_or_err(path_type.path(), &path_type, "path type");
                    ast::TyKind::Path(qself.map(|(qself, _)| qself), path)
                }
                ra::TypeRef::NeverType(_) => ast::TyKind::Never,
                ra::TypeRef::PlaceholderType(_) => ast::TyKind::Infer,
                ra::TypeRef::TupleType(tuple_type) => {
                    ast::TyKind::Tup(tuple_type.fields().map(|field| self.to_ty(field)).collect())
                }
                ra::TypeRef::ArrayType(array_type) => ast::TyKind::Array(
                    self.to_ty_or_err(array_type.type_ref(), &array_type, "array type"),
                    self.to_anon_const_or_err(array_type.expr(), &array_type, "array type size"),
                ),
                ra::TypeRef::SliceType(slice_type) => ast::TyKind::Slice(self.to_ty_or_err(
                    slice_type.type_ref(),
                    &slice_type,
                    "slice type",
                )),
                ra::TypeRef::PointerType(pointer_type) => ast::TyKind::Ptr(ast::MutTy {
                    ty: self.to_ty_or_err(pointer_type.type_ref(), &pointer_type, "pointer type"),
                    mutbl: self.to_mutability_from_const(pointer_type.const_kw()),
                }),
                ra::TypeRef::ReferenceType(reference_type) => ast::TyKind::Rptr(
                    reference_type.lifetime().map(|lifetime| self.to_lifetime(lifetime)),
                    ast::MutTy {
                        ty: self.to_ty_or_err(
                            reference_type.type_ref(),
                            &reference_type,
                            "reference type",
                        ),
                        mutbl: self.to_mutability(reference_type.mut_kw()),
                    },
                ),
                ra::TypeRef::ParenType(paren_type) => ast::TyKind::Paren(self.to_ty_or_err(
                    paren_type.type_ref(),
                    &paren_type,
                    "parenthesized type",
                )),
                ra::TypeRef::DynTraitType(dyn_trait_type) => ast::TyKind::TraitObject(
                    self.to_generic_bounds(dyn_trait_type.type_bound_list()),
                    if dyn_trait_type.dyn_kw().is_some() {
                        ast::TraitObjectSyntax::Dyn
                    } else {
                        ast::TraitObjectSyntax::None
                    },
                ),
                ra::TypeRef::ImplTraitType(impl_trait_type) => ast::TyKind::ImplTrait(
                    ast::DUMMY_NODE_ID,
                    self.to_generic_bounds(impl_trait_type.type_bound_list()),
                ),
                ra::TypeRef::FnPointerType(fn_pointer_type) => {
                    ast::TyKind::BareFn(self.to_bare_fn_ty(fn_pointer_type, None))
                }
                ra::TypeRef::ForType(for_type) => {
                    if let Some(inner_type_ref) = for_type.type_ref() {
                        match inner_type_ref {
                            ra::TypeRef::FnPointerType(fn_pointer_type) => ast::TyKind::BareFn(
                                self.to_bare_fn_ty(fn_pointer_type, for_type.type_param_list()),
                            ),
                            inner_type_ref => {
                                // other cases, only allowed in where clauses, are handled by special casing the for-type
                                self.missing_in(&for_type, "fn() type", "for<> type");
                                type_ref = inner_type_ref;
                                continue;
                            }
                        }
                    } else {
                        self.missing_in(&for_type, "inner type", "for<> type");
                        ast::TyKind::Err
                    }
                }
            };
            return P(ast::Ty { id: ast::DUMMY_NODE_ID, span, kind });
        }
    }

    fn to_lifetime_or_err(
        &self,
        lifetime_opt: Option<ra::Lifetime>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> ast::Lifetime {
        self.expect_in(lifetime_opt, element, "lifetime", container)
            .map(|lifetime| self.to_lifetime(lifetime))
            .unwrap_or_else(|| ast::Lifetime {
                id: ast::DUMMY_NODE_ID,
                ident: ast::Ident::new(kw::Invalid, self.get_inner_span(element)),
            })
    }

    fn to_lifetime(&self, lifetime: ra::Lifetime) -> ast::Lifetime {
        ast::Lifetime { id: ast::DUMMY_NODE_ID, ident: self.to_ident_from_token(lifetime) }
    }

    fn to_label(&self, label: ra::Label) -> ast::Label {
        ast::Label { ident: self.to_lifetime_or_err(label.lifetime(), &label, "label").ident }
    }

    fn to_label_from_lifetime(&self, lifetime: ra::Lifetime) -> ast::Label {
        ast::Label { ident: self.to_lifetime(lifetime).ident }
    }

    fn to_mac_call(&self, macro_call: ra::MacroCall) -> ast::MacCall {
        self.invalid_element(
            macro_call.name(),
            &macro_call,
            "name",
            "macro call other than macro_rules!",
        );
        ast::MacCall {
            path: self.to_simple_path_or_err(macro_call.path(), &macro_call, "invoked macro name"),
            args: P(if let Some(token_tree) = macro_call.token_tree() {
                self.to_mac_args_from_token_tree(token_tree)
            } else {
                self.error(&macro_call, "macro calls must have arguments");
                ast::MacArgs::Empty
            }),
            // TODO: this is just used for a better error in a very niche case, maybe we should somehow support it anyway
            prior_type_ascription: None,
        }
    }

    fn mk_path_for_ident(&self, ident: ast::Ident) -> ast::Path {
        ast::Path {
            span: ident.span,
            segments: vec![ast::PathSegment {
                id: ast::DUMMY_NODE_ID,
                args: Default::default(),
                ident,
            }],
        }
    }

    fn mk_expr_for_ident(&self, ident: ast::Ident) -> P<ast::Expr> {
        P(ast::Expr {
            id: ast::DUMMY_NODE_ID,
            attrs: Default::default(),
            span: ident.span,
            kind: ast::ExprKind::Path(None, self.mk_path_for_ident(ident)),
        })
    }

    fn to_expr(&self, expr: ra::Expr) -> P<ast::Expr> {
        self.to_expr_from_expr_and_expr_stmt_opt(expr, None)
    }

    fn to_expr_from_expr_and_expr_stmt_opt(
        &self,
        expr: ra::Expr,
        expr_stmt: Option<ra::ExprStmt>,
    ) -> P<ast::Expr> {
        let span = self.get_inner_span(&expr);
        let attrs = expr_stmt
            .iter()
            .flat_map(|expr_stmt| self.get_attributes(expr_stmt))
            .chain(self.get_attributes(&expr))
            .collect::<Vec<_>>()
            .into();
        let kind = match expr {
            ra::Expr::ArrayExpr(array_expr) => {
                if let Some(_semi) = array_expr.semi() {
                    let mut exprs = array_expr.exprs();
                    let expr_kind = ast::ExprKind::Repeat(
                        self.to_expr_or_err(
                            exprs.next(),
                            &array_expr,
                            "array repeat expression value",
                        ),
                        self.to_anon_const_or_err(
                            exprs.next(),
                            &array_expr,
                            "array repeat expression repetition count",
                        ),
                    );
                    self.invalid_element(
                        exprs.next(),
                        &array_expr,
                        "3rd expression",
                        "array repeat expression",
                    );
                    expr_kind
                } else {
                    ast::ExprKind::Array(
                        array_expr.exprs().map(|expr| self.to_expr(expr)).collect(),
                    )
                }
            }
            ra::Expr::AwaitExpr(await_expr) => ast::ExprKind::Await(self.to_expr_or_err(
                await_expr.expr(),
                &await_expr,
                "await expression",
            )),
            ra::Expr::BinExpr(bin_expr) => {
                if let Some(bin_op) = self.expect_in(
                    bin_expr.bin_op(),
                    &bin_expr,
                    "recognized binary operator",
                    "binary expression",
                ) {
                    let span = self.get_inner_span(&bin_op);
                    let lhs = self.to_expr_or_err(
                        bin_expr.lhs(),
                        &bin_expr,
                        "left-hand side of binary operator",
                    );
                    let rhs = self.to_expr_or_err(
                        bin_expr.rhs(),
                        &bin_expr,
                        "right-hand side of binary operator",
                    );
                    enum Kind {
                        Assign,
                        AssignOp,
                        Binary,
                    };

                    // TODO: maybe rewrite to match bin_op directly
                    let (node, kind) = match bin_expr.op_kind().unwrap() {
                        ra::BinOp::Assignment => (ast::BinOpKind::Eq, Kind::Assign),

                        ra::BinOp::AddAssign => (ast::BinOpKind::Add, Kind::AssignOp),
                        ra::BinOp::BitAndAssign => (ast::BinOpKind::BitAnd, Kind::AssignOp),
                        ra::BinOp::BitOrAssign => (ast::BinOpKind::BitOr, Kind::AssignOp),
                        ra::BinOp::BitXorAssign => (ast::BinOpKind::BitXor, Kind::AssignOp),
                        ra::BinOp::DivAssign => (ast::BinOpKind::Div, Kind::AssignOp),
                        ra::BinOp::MulAssign => (ast::BinOpKind::Mul, Kind::AssignOp),
                        ra::BinOp::RemAssign => (ast::BinOpKind::Rem, Kind::AssignOp),
                        ra::BinOp::ShlAssign => (ast::BinOpKind::Shl, Kind::AssignOp),
                        ra::BinOp::ShrAssign => (ast::BinOpKind::Shr, Kind::AssignOp),
                        ra::BinOp::SubAssign => (ast::BinOpKind::Sub, Kind::AssignOp),

                        ra::BinOp::Addition => (ast::BinOpKind::Add, Kind::Binary),
                        ra::BinOp::BitwiseAnd => (ast::BinOpKind::BitAnd, Kind::Binary),
                        ra::BinOp::BitwiseOr => (ast::BinOpKind::BitOr, Kind::Binary),
                        ra::BinOp::BitwiseXor => (ast::BinOpKind::BitXor, Kind::Binary),
                        ra::BinOp::BooleanAnd => (ast::BinOpKind::And, Kind::Binary),
                        ra::BinOp::BooleanOr => (ast::BinOpKind::Or, Kind::Binary),
                        ra::BinOp::Division => (ast::BinOpKind::Div, Kind::Binary),
                        ra::BinOp::EqualityTest => (ast::BinOpKind::Eq, Kind::Binary),
                        ra::BinOp::GreaterEqualTest => (ast::BinOpKind::Ge, Kind::Binary),
                        ra::BinOp::GreaterTest => (ast::BinOpKind::Gt, Kind::Binary),
                        ra::BinOp::LeftShift => (ast::BinOpKind::Shl, Kind::Binary),
                        ra::BinOp::LesserEqualTest => (ast::BinOpKind::Le, Kind::Binary),
                        ra::BinOp::LesserTest => (ast::BinOpKind::Lt, Kind::Binary),
                        ra::BinOp::Multiplication => (ast::BinOpKind::Mul, Kind::Binary),
                        ra::BinOp::NegatedEqualityTest => (ast::BinOpKind::Ne, Kind::Binary),
                        ra::BinOp::Remainder => (ast::BinOpKind::Rem, Kind::Binary),
                        ra::BinOp::RightShift => (ast::BinOpKind::Shr, Kind::Binary),
                        ra::BinOp::Subtraction => (ast::BinOpKind::Sub, Kind::Binary),
                    };

                    match kind {
                        Kind::Assign => ast::ExprKind::Assign(lhs, rhs, span),
                        Kind::AssignOp => ast::ExprKind::AssignOp(Spanned { span, node }, lhs, rhs),
                        Kind::Binary => ast::ExprKind::Binary(Spanned { span, node }, lhs, rhs),
                    }
                } else {
                    ast::ExprKind::Err
                }
            }
            ra::Expr::BlockExpr(block_expr) => {
                let label = block_expr.label().map(|label| self.to_label(label));
                ast::ExprKind::Block(self.to_block_from_block_expr(block_expr), label)
            }
            ra::Expr::BoxExpr(box_expr) => ast::ExprKind::Box(self.to_expr_or_err(
                box_expr.expr(),
                &box_expr,
                "box expression",
            )),
            ra::Expr::BreakExpr(break_expr) => ast::ExprKind::Break(
                break_expr.lifetime().map(|lifetime| self.to_label_from_lifetime(lifetime)),
                break_expr.expr().map(|expr| self.to_expr(expr)),
            ),
            ra::Expr::CallExpr(call_expr) => ast::ExprKind::Call(
                self.to_expr_or_err(call_expr.expr(), &call_expr, "function to be called"),
                call_expr
                    .arg_list()
                    .into_iter()
                    .flat_map(|arg_list| arg_list.args().map(|expr| self.to_expr(expr)))
                    .collect(),
            ),
            ra::Expr::CastExpr(cast_expr) => ast::ExprKind::Cast(
                self.to_expr_or_err(cast_expr.expr(), &cast_expr, "cast expression"),
                self.to_ty_or_err(cast_expr.type_ref(), &cast_expr, "cast expression"),
            ),
            ra::Expr::ContinueExpr(continue_expr) => ast::ExprKind::Continue(
                continue_expr.lifetime().map(|lifetime| self.to_label_from_lifetime(lifetime)),
            ),
            ra::Expr::FieldExpr(field_expr) => ast::ExprKind::Field(
                self.to_expr_or_err(field_expr.expr(), &field_expr, "field access expression"),
                self.to_ident_from_name_ref_or_err(
                    field_expr.name_ref(),
                    &field_expr,
                    "field access expression",
                ),
            ),
            ra::Expr::ForExpr(for_expr) => ast::ExprKind::ForLoop(
                self.to_pat_or_err(for_expr.pat(), &for_expr, "for loop"),
                self.to_expr_or_err(for_expr.iterable(), &for_expr, "for loop"),
                self.to_block_or_err(
                    for_expr.loop_body().and_then(|block_expr| block_expr.block()),
                    &for_expr,
                    "for loop",
                ),
                for_expr.label().map(|label| self.to_label(label)),
            ),
            ra::Expr::IfExpr(if_expr) => ast::ExprKind::If(
                self.to_expr_from_condition_or_err(if_expr.condition(), &if_expr, "if conditional"),
                self.to_block_or_err(
                    if_expr.then_branch().and_then(|block_expr| block_expr.block()),
                    &if_expr,
                    "if conditional",
                ),
                if_expr.else_branch().map(|else_branch| match else_branch {
                    ra::ElseBranch::Block(else_expr) => {
                        self.to_expr(ra::Expr::BlockExpr(else_expr))
                    }
                    ra::ElseBranch::IfExpr(else_if_expr) => {
                        self.to_expr(ra::Expr::IfExpr(else_if_expr))
                    }
                }),
            ),
            ra::Expr::IndexExpr(index_expr) => ast::ExprKind::Index(
                self.to_expr_or_err(index_expr.base(), &index_expr, "index expression base"),
                self.to_expr_or_err(index_expr.index(), &index_expr, "index expression index"),
            ),
            ra::Expr::Label(label) => {
                self.error(&label, "unexpected label expression in generic context");
                ast::ExprKind::Err
            }
            ra::Expr::LambdaExpr(lambda_expr) => {
                let end = if let Some(body) = lambda_expr.body() {
                    inner_start(&body)
                } else {
                    start_of_next(&lambda_expr)
                };
                ast::ExprKind::Closure(
                    if lambda_expr.move_kw().is_some() {
                        ast::CaptureBy::Value
                    } else {
                        ast::CaptureBy::Ref
                    },
                    self.to_async(lambda_expr.async_kw()),
                    if lambda_expr.static_kw().is_some() {
                        ast::Movability::Static
                    } else {
                        ast::Movability::Movable
                    },
                    self.to_fn_decl(
                        lambda_expr.param_list(),
                        lambda_expr.ret_type(),
                        end,
                        FnParamMode { require_pat: true, require_type: false },
                    ),
                    self.to_expr_or_err(lambda_expr.body(), &lambda_expr, "closure body"),
                    if let Some(param_list) = lambda_expr.param_list() {
                        self.get_inner_span(&param_list)
                    } else {
                        self.get_inner_span(&lambda_expr)
                    },
                )
            }
            ra::Expr::Literal(literal) => ast::ExprKind::Lit(self.to_lit_or_err(
                literal.literal_token(),
                &literal,
                "literal expression",
            )),
            ra::Expr::LoopExpr(loop_expr) => ast::ExprKind::Loop(
                self.to_block_or_err(
                    loop_expr.loop_body().and_then(|block_expr| block_expr.block()),
                    &loop_expr,
                    "loop",
                ),
                loop_expr.label().map(|label| self.to_label(label)),
            ),
            ra::Expr::MacroCall(macro_call) => ast::ExprKind::MacCall(self.to_mac_call(macro_call)),
            ra::Expr::MatchExpr(match_expr) => ast::ExprKind::Match(
                self.to_expr_or_err(match_expr.expr(), &match_expr, "match selector"),
                if let Some(match_arm_list) = match_expr.match_arm_list() {
                    match_arm_list
                        .arms()
                        .map(|match_arm| ast::Arm {
                            id: ast::DUMMY_NODE_ID,
                            span: self.get_inner_to_next_span(&match_arm),
                            attrs: self.get_attributes(&match_arm).collect(),
                            pat: self.to_pat_or_err(
                                match_arm.pat(),
                                &match_arm,
                                "match arm condition",
                            ),
                            guard: match_arm.guard().map(|guard| {
                                self.to_expr_or_err(guard.expr(), &guard, "match guard")
                            }),
                            body: self.to_expr_or_err(
                                match_arm.expr(),
                                &match_arm,
                                "match arm body",
                            ),
                            is_placeholder: false,
                        })
                        .collect()
                } else {
                    Default::default()
                },
            ),
            ra::Expr::MethodCallExpr(method_call_expr) => ast::ExprKind::MethodCall(
                ast::PathSegment {
                    id: ast::DUMMY_NODE_ID,
                    ident: self.to_ident_from_name_ref_or_err(
                        method_call_expr.name_ref(),
                        &method_call_expr,
                        "method name",
                    ),
                    args: if let Some(type_arg_list) = method_call_expr.type_arg_list() {
                        Some(P(ast::GenericArgs::AngleBracketed(
                            self.to_angle_bracketed_args(&type_arg_list),
                        )))
                    } else {
                        None
                    },
                },
                Some(self.to_expr_or_err(
                    method_call_expr.expr(),
                    &method_call_expr,
                    "method call receiver",
                ))
                .into_iter()
                .chain(
                    method_call_expr
                        .arg_list()
                        .into_iter()
                        .flat_map(|arg_list| arg_list.args().map(|expr| self.to_expr(expr))),
                )
                .collect(),
            ),
            ra::Expr::ParenExpr(paren_expr) => ast::ExprKind::Paren(self.to_expr_or_err(
                paren_expr.expr(),
                &paren_expr,
                "parenthesized expression",
            )),
            ra::Expr::PathExpr(path_expr) => {
                let (qself, path) =
                    self.to_qself_and_path_or_err(path_expr.path(), &path_expr, "path expression");
                ast::ExprKind::Path(qself.map(|(qself, _)| qself), path)
            }
            ra::Expr::PrefixExpr(prefix_expr) => {
                if let Some(op_kind) = self.expect_in(
                    prefix_expr.op_kind(),
                    &prefix_expr,
                    "recognized prefix operator",
                    "prefix operator expression",
                ) {
                    ast::ExprKind::Unary(
                        match op_kind {
                            ra::PrefixOp::Deref => ast::UnOp::Deref,
                            ra::PrefixOp::Neg => ast::UnOp::Neg,
                            ra::PrefixOp::Not => ast::UnOp::Not,
                        },
                        self.to_expr_or_err(
                            prefix_expr.expr(),
                            &prefix_expr,
                            "prefix operator expression",
                        ),
                    )
                } else {
                    ast::ExprKind::Err
                }
            }
            ra::Expr::RangeExpr(range_expr) => {
                if let Some(op_kind) = self.expect_in(
                    range_expr.op_kind(),
                    &range_expr,
                    "recognized range separator",
                    "range expression",
                ) {
                    ast::ExprKind::Range(
                        range_expr.start().map(|start| self.to_expr(start)),
                        range_expr.end().map(|end| self.to_expr(end)),
                        match op_kind {
                            ra::RangeOp::Exclusive => ast::RangeLimits::HalfOpen,
                            ra::RangeOp::Inclusive => ast::RangeLimits::Closed,
                        },
                    )
                } else {
                    ast::ExprKind::Err
                }
            }
            ra::Expr::RecordLit(record_lit) => ast::ExprKind::Struct(
                self.to_simple_path_or_err(record_lit.path(), &record_lit, "struct initializer"),
                if let Some(record_field_list) = record_lit.record_field_list() {
                    record_field_list
                        .fields()
                        .map(|field| {
                            let ident = self.to_ident_from_name_ref_or_err(
                                field.name_ref(),
                                &field,
                                "record field initialization",
                            );
                            ast::Field {
                                id: ast::DUMMY_NODE_ID,
                                span: self.get_inner_span(&field),
                                attrs: self.get_attributes(&field).collect::<Vec<_>>().into(),
                                ident,
                                expr: if let Some(expr) = field.expr() {
                                    self.to_expr(expr)
                                } else {
                                    self.mk_expr_for_ident(ident)
                                },
                                is_shorthand: field.expr().is_none(),
                                is_placeholder: false,
                            }
                        })
                        .collect()
                } else {
                    Default::default()
                },
                if let Some(record_field_list) = record_lit.record_field_list() {
                    record_field_list.spread().map(|expr| self.to_expr(expr))
                } else {
                    None
                },
            ),
            ra::Expr::RefExpr(ref_expr) => ast::ExprKind::AddrOf(
                self.to_borrow_kind(ref_expr.raw_kw()),
                self.to_mutability(ref_expr.mut_kw()),
                self.to_expr_or_err(ref_expr.expr(), &ref_expr, "reference expression"),
            ),
            ra::Expr::ReturnExpr(return_expr) => {
                ast::ExprKind::Ret(return_expr.expr().map(|value_expr| self.to_expr(value_expr)))
            }
            ra::Expr::TryBlockExpr(try_block_expr) => {
                ast::ExprKind::TryBlock(self.to_block_or_err(
                    try_block_expr.body().and_then(|block_expr| block_expr.block()),
                    &try_block_expr,
                    "try block",
                ))
            }
            ra::Expr::TryExpr(try_expr) => ast::ExprKind::Try(self.to_expr_or_err(
                try_expr.expr(),
                &try_expr,
                "error propagation expression",
            )),
            ra::Expr::TupleExpr(tuple_expr) => ast::ExprKind::Tup(
                tuple_expr.exprs().map(|inner_expr| self.to_expr(inner_expr)).collect(),
            ),
            ra::Expr::WhileExpr(while_expr) => ast::ExprKind::While(
                self.to_expr_from_condition_or_err(
                    while_expr.condition(),
                    &while_expr,
                    "while loop",
                ),
                self.to_block_or_err(
                    while_expr.loop_body().and_then(|block_expr| block_expr.block()),
                    &while_expr,
                    "while loop",
                ),
                while_expr.label().map(|label| self.to_label(label)),
            ),
        };
        P(ast::Expr { id: ast::DUMMY_NODE_ID, span, attrs, kind })
    }

    fn to_expr_or_err(
        &self,
        expr_opt: Option<ra::Expr>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> P<ast::Expr> {
        self.expect_in(expr_opt, element, "expression", container)
            .map(|expr| self.to_expr(expr))
            .unwrap_or_else(|| {
                P(ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    span: self.get_inner_span(element),
                    attrs: Default::default(),
                    kind: ast::ExprKind::Err,
                })
            })
    }

    fn get_item_kind_opt_from_macro_call(
        &self,
        macro_call: &ra::MacroCall,
    ) -> Option<ast::ItemKind> {
        if let Some(path) = macro_call.path() {
            if path.qualifier().is_none() {
                if let Some(segment) = path.segment() {
                    if let Some(name_ref) = segment.name_ref() {
                        if let Some(ra::NameRefToken::Ident(ident)) = name_ref.name_ref_token() {
                            if ident.text() == "macro_rules" {
                                return Some(ast::ItemKind::MacroDef(ast::MacroDef {
                                    body: P(if let Some(token_tree) = macro_call.token_tree() {
                                        self.to_mac_args_from_token_tree(token_tree)
                                    } else {
                                        self.error(
                                            macro_call,
                                            "macro_rules definitions must have arguments",
                                        );
                                        ast::MacArgs::Empty
                                    }),
                                    macro_rules: true,
                                }));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn to_stmt_kind_from_macro_call(
        &self,
        macro_call: ra::MacroCall,
        force_mac_stmt_style_opt: Option<ast::MacStmtStyle>,
    ) -> ast::StmtKind {
        if let Some(item_kind) = self.get_item_kind_opt_from_macro_call(&macro_call) {
            let span = self.get_inner_span(&macro_call);
            ast::StmtKind::Item(P(ast::Item {
                id: ast::DUMMY_NODE_ID,
                span,
                attrs: self.get_attributes(&macro_call).collect(),
                vis: Spanned { span: span.shrink_to_lo(), node: ast::VisibilityKind::Inherited },
                ident: self.to_ident_from_name_or_err(
                    macro_call.name(),
                    &macro_call,
                    "macro_rules! macro name",
                ),
                kind: item_kind,
                tokens: self.get_collected_tokens_from_range(inner_range(&macro_call)),
            }))
        } else {
            let mac_stmt_style = if let Some(force_mac_stmt_style) = force_mac_stmt_style_opt {
                force_mac_stmt_style
            } else if macro_call.semi().is_some() {
                ast::MacStmtStyle::Semicolon
            } else if macro_call
                .token_tree()
                .filter(|x| match x.left_delimiter() {
                    Some(ra::LeftDelimiter::LCurly(_)) => true,
                    _ => false,
                })
                .is_some()
            {
                ast::MacStmtStyle::Braces
            } else {
                ast::MacStmtStyle::NoBraces
            };
            let attrs = self.get_attributes(&macro_call).collect::<Vec<_>>().into();

            if mac_stmt_style == ast::MacStmtStyle::NoBraces {
                ast::StmtKind::Expr(self.to_expr(ra::Expr::MacroCall(macro_call)))
            } else {
                ast::StmtKind::MacCall(P((self.to_mac_call(macro_call), mac_stmt_style, attrs)))
            }
        }
    }

    fn to_stmt(&self, stmt: ra::StmtOrSemi) -> ast::Stmt {
        ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            span: self.get_inner_span(&stmt),
            kind: match stmt {
                ra::StmtOrSemi::Stmt(ra::Stmt::ExprStmt(expr_stmt)) => {
                    let semi = expr_stmt.semi();
                    match expr_stmt.expr() {
                        Some(ra::Expr::MacroCall(macro_call)) => self.to_stmt_kind_from_macro_call(
                            macro_call,
                            Some(ast::MacStmtStyle::Semicolon),
                        ),
                        expr => {
                            let ast_expr = if let Some(expr) = self.expect_in(
                                expr,
                                &expr_stmt,
                                "expression",
                                "expression statement",
                            ) {
                                self.to_expr_from_expr_and_expr_stmt_opt(expr, Some(expr_stmt))
                            } else {
                                P(ast::Expr {
                                    id: ast::DUMMY_NODE_ID,
                                    span: self.get_inner_span(&expr_stmt),
                                    attrs: Default::default(),
                                    kind: ast::ExprKind::Err,
                                })
                            };
                            if semi.is_some() {
                                ast::StmtKind::Semi(ast_expr)
                            } else {
                                ast::StmtKind::Expr(ast_expr)
                            }
                        }
                    }
                }
                ra::StmtOrSemi::Stmt(ra::Stmt::LetStmt(let_stmt)) => {
                    ast::StmtKind::Local(P(ast::Local {
                        id: ast::DUMMY_NODE_ID,
                        span: self.get_inner_span(&let_stmt),
                        attrs: self.get_attributes(&let_stmt).collect::<Vec<_>>().into(),
                        pat: self.to_pat_or_err(let_stmt.pat(), &let_stmt, "let statement"),
                        ty: let_stmt.ascribed_type().map(|type_ref| self.to_ty(type_ref)),
                        init: let_stmt.initializer().map(|expr| self.to_expr(expr)),
                    }))
                }
                ra::StmtOrSemi::Stmt(ra::Stmt::ModuleItem(module_item)) => {
                    ast::StmtKind::Item(self.to_module_item(module_item, "code block"))
                }
                ra::StmtOrSemi::Semi(_) => ast::StmtKind::Empty,
            },
        }
    }

    fn to_block(&self, block: ra::Block) -> P<ast::Block> {
        self.to_block_from_block_and_block_expr_opt(block, None)
    }

    fn to_block_from_block_and_block_expr_opt(
        &self,
        block: ra::Block,
        block_expr: Option<ra::BlockExpr>,
    ) -> P<ast::Block> {
        P(ast::Block {
            id: ast::DUMMY_NODE_ID,
            span: if let Some(block_expr) = &block_expr {
                self.get_inner_span(block_expr)
            } else {
                self.get_inner_span(&block)
            },
            rules: if block_expr.and_then(|block_expr| block_expr.unsafe_kw()).is_some() {
                // compiler generated seems to only be used in derive expansions
                ast::BlockCheckMode::Unsafe(ast::UnsafeSource::UserProvided)
            } else {
                ast::BlockCheckMode::Default
            },
            stmts: block
                .statements_or_semi()
                .filter_map(|stmt| match stmt {
                    ra::StmtOrSemi::Stmt(ra::Stmt::ModuleItem(ra::ModuleItem::MacroCall(_))) => {
                        // TODO: HACK: this is actually a macro call as the final block expression, which is wrongly included in statements_or_semi
                        None
                    }
                    stmt => Some(self.to_stmt(stmt)),
                })
                .chain(block.expr().into_iter().map(|expr| ast::Stmt {
                    id: ast::DUMMY_NODE_ID,
                    span: self.get_inner_span(&expr),
                    kind: match expr {
                        ra::Expr::MacroCall(macro_call) => {
                            self.to_stmt_kind_from_macro_call(macro_call, None)
                        }
                        expr => ast::StmtKind::Expr(self.to_expr(expr)),
                    },
                }))
                .collect(),
        })
    }

    fn to_block_or_err(
        &self,
        block_opt: Option<ra::Block>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> P<ast::Block> {
        if let Some(block) = self.expect_in(block_opt, element, "block", container) {
            self.to_block_from_block_and_block_expr_opt(block, None)
        } else {
            P(ast::Block {
                id: ast::DUMMY_NODE_ID,
                span: self.get_inner_span(element),
                rules: ast::BlockCheckMode::Default,
                stmts: Default::default(),
            })
        }
    }

    fn to_block_from_block_expr(&self, block_expr: ra::BlockExpr) -> P<ast::Block> {
        if let Some(block) =
            self.expect_in(block_expr.block(), &block_expr, "block", "block expression")
        {
            self.to_block_from_block_and_block_expr_opt(block, Some(block_expr))
        } else {
            P(ast::Block {
                id: ast::DUMMY_NODE_ID,
                span: self.get_inner_span(&block_expr),
                rules: if block_expr.unsafe_kw().is_some() {
                    // compiler generated seems to only be used in derive expansions
                    ast::BlockCheckMode::Unsafe(ast::UnsafeSource::UserProvided)
                } else {
                    ast::BlockCheckMode::Default
                },
                stmts: Default::default(),
            })
        }
    }

    fn to_anon_const(&self, expr: ra::Expr) -> ast::AnonConst {
        ast::AnonConst { id: ast::DUMMY_NODE_ID, value: self.to_expr(expr) }
    }

    fn to_anon_const_or_err(
        &self,
        expr_opt: Option<ra::Expr>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> ast::AnonConst {
        ast::AnonConst {
            id: ast::DUMMY_NODE_ID,
            value: self.to_expr_or_err(expr_opt, element, container),
        }
    }

    fn to_assoc_ty_constraint(&self, assoc_type_arg: ra::AssocTypeArg) -> ast::AssocTyConstraint {
        ast::AssocTyConstraint {
            id: ast::DUMMY_NODE_ID,
            span: self.get_inner_span(&assoc_type_arg),
            ident: self.to_ident_from_name_ref_or_err(
                assoc_type_arg.name_ref(),
                &assoc_type_arg,
                "associated type constraint",
            ),
            kind: if let Some(type_ref) = assoc_type_arg.type_ref() {
                ast::AssocTyConstraintKind::Equality { ty: self.to_ty(type_ref) }
            } else {
                ast::AssocTyConstraintKind::Bound {
                    bounds: self.to_generic_bounds(assoc_type_arg.type_bound_list()),
                }
            },
        }
    }

    fn to_angle_bracketed_args(&self, type_arg_list: &ra::TypeArgList) -> ast::AngleBracketedArgs {
        ast::AngleBracketedArgs {
            span: self.to_span(TextRange::from_to(
                if let Some(coloncolon) = type_arg_list.coloncolon() {
                    start_of_next(&coloncolon)
                } else {
                    inner_start(type_arg_list)
                },
                inner_end(type_arg_list),
            )),
            args: type_arg_list
                .generic_args()
                .map(|arg| match arg {
                    ra::GenericArg::TypeArg(type_arg) => {
                        ast::AngleBracketedArg::Arg(ast::GenericArg::Type(self.to_ty_or_err(
                            type_arg.type_ref(),
                            &type_arg,
                            "type argument",
                        )))
                    }
                    ra::GenericArg::LifetimeArg(lifetime_arg) => ast::AngleBracketedArg::Arg(
                        ast::GenericArg::Lifetime(self.to_lifetime_or_err(
                            lifetime_arg.lifetime(),
                            &lifetime_arg,
                            "lifetime argument",
                        )),
                    ),
                    ra::GenericArg::ConstArg(const_arg) => {
                        ast::AngleBracketedArg::Arg(ast::GenericArg::Const(
                            self.to_anon_const_or_err(
                                const_arg
                                    .block_expr()
                                    .map(|block_expr| ra::Expr::BlockExpr(block_expr)),
                                &const_arg,
                                "constant generic argument",
                            ),
                        ))
                    }
                    ra::GenericArg::AssocTypeArg(assoc_type_arg) => {
                        ast::AngleBracketedArg::Constraint(
                            self.to_assoc_ty_constraint(assoc_type_arg),
                        )
                    }
                })
                .collect(),
        }
    }

    fn get_generics_for_named(
        &self,
        element: &(impl TypeParamsOwner + NameOwner),
        type_param_list_end: impl FnOnce() -> Option<TextUnit>,
        where_clause_end: impl FnOnce() -> Option<TextUnit>,
    ) -> ast::Generics {
        self.get_generics_for_unnamed(
            element,
            || {
                if let Some(name) = element.name() {
                    Some(inner_end(&name))
                } else {
                    type_param_list_end()
                }
            },
            where_clause_end,
        )
    }

    fn get_generics_for_unnamed(
        &self,
        element: &impl ra::TypeParamsOwner,
        type_param_list_end: impl FnOnce() -> Option<TextUnit>,
        where_clause_end: impl FnOnce() -> Option<TextUnit>,
    ) -> ast::Generics {
        self.to_generics(
            element.type_param_list().ok_or_else(|| {
                if let Some(type_param_list_end) = type_param_list_end() {
                    type_param_list_end
                } else {
                    inner_start(element)
                }
            }),
            element.where_clause().ok_or_else(|| {
                if let Some(where_clause_end) = where_clause_end() {
                    where_clause_end
                } else {
                    start_of_next(element)
                }
            }),
        )
    }

    fn to_generics(
        &self,
        type_param_list: Result<ra::TypeParamList, TextUnit>,
        where_clause: Result<ra::WhereClause, TextUnit>,
    ) -> ast::Generics {
        ast::Generics {
            span: match type_param_list.as_ref() {
                Ok(type_param_list) => self.get_inner_span(type_param_list),
                Err(insert_type_param_list) => self.to_span_from_unit(*insert_type_param_list),
            },
            params: self.to_generic_params(type_param_list.ok()),
            where_clause: match where_clause {
                Ok(where_clause) => self.to_where_clause(where_clause),
                Err(insert_where_clause) => ast::WhereClause {
                    span: self.to_span_from_unit(insert_where_clause),
                    predicates: Default::default(),
                },
            },
        }
    }

    fn unwrap_for_type(
        &self,
        type_ref_opt: Option<ra::TypeRef>,
    ) -> (Option<ra::TypeRef>, Option<ra::TypeParamList>) {
        match type_ref_opt {
            Some(ra::TypeRef::ForType(for_type)) => {
                (for_type.type_ref(), for_type.type_param_list())
            }
            type_ref_opt => (type_ref_opt, None),
        }
    }

    fn to_where_clause(&self, node: ra::WhereClause) -> ast::WhereClause {
        ast::WhereClause {
            span: self.get_inner_span(&node),
            predicates: node
                .predicates()
                .map(|pred: ra::WherePred| {
                    if let Some(lifetime) = pred.lifetime() {
                        ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate {
                            span: self.get_inner_span(&pred),
                            lifetime: self.to_lifetime(lifetime),
                            bounds: self.get_generic_bounds(&pred),
                        })
                    } else {
                        let (type_ref_opt, type_param_list_opt) =
                            self.unwrap_for_type(pred.type_ref());
                        ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate {
                            span: self.get_inner_span(&pred),
                            bound_generic_params: self.to_generic_params(type_param_list_opt),
                            bounded_ty: self.to_ty_or_err(type_ref_opt, &pred, "where predicate"),
                            bounds: self.get_generic_bounds(&pred),
                        })
                    }
                })
                .collect(),
        }
    }

    fn get_generic_bounds(&self, node: &impl TypeBoundsOwner) -> ast::GenericBounds {
        self.to_generic_bounds(node.type_bound_list())
    }

    fn to_generic_bounds(&self, node: Option<ra::TypeBoundList>) -> ast::GenericBounds {
        if let Some(bounds) = node {
            bounds
                .bounds()
                .map(|bound: ra::TypeBound| {
                    let bound_mod = if bound.const_kw().is_some() {
                        self.expect_in(bound.const_question(), &bound, "?", "? const bound");
                        if bound.question().is_some() {
                            ast::TraitBoundModifier::MaybeConstMaybe
                        } else {
                            ast::TraitBoundModifier::MaybeConst
                        }
                    } else {
                        if bound.question().is_some() {
                            ast::TraitBoundModifier::Maybe
                        } else {
                            ast::TraitBoundModifier::None
                        }
                    };
                    match bound.kind() {
                        ra::TypeBoundKind::Lifetime(lifetime) => {
                            if bound_mod != ast::TraitBoundModifier::None {
                                self.error(&bound, "? modifier on lifetime bound");
                            }
                            ast::GenericBound::Outlives(self.to_lifetime(lifetime))
                        }
                        ra::TypeBoundKind::PathType(path_type) => ast::GenericBound::Trait(
                            ast::PolyTraitRef::new(
                                Default::default(),
                                self.to_trait_path_from_path_type(path_type),
                                self.get_inner_span(&bound),
                            ),
                            bound_mod,
                        ),
                        ra::TypeBoundKind::ForType(for_type) => self
                            .expect_in(for_type.type_ref(), &for_type, "type", "for type")
                            .and_then(|type_ref| {
                                self.expect_in_res(
                                    match type_ref {
                                        ra::TypeRef::PathType(path_type) => Ok(path_type),
                                        x => Err(x),
                                    },
                                    "path type",
                                    "for type",
                                )
                                .map(|path_type| {
                                    ast::GenericBound::Trait(
                                        ast::PolyTraitRef::new(
                                            self.to_generic_params(for_type.type_param_list()),
                                            self.to_trait_path_from_path_type(path_type),
                                            self.get_inner_span(&bound),
                                        ),
                                        bound_mod,
                                    )
                                })
                            })
                            .unwrap_or_else(|| {
                                ast::GenericBound::Trait(
                                    ast::PolyTraitRef::new(
                                        self.to_generic_params(for_type.type_param_list()),
                                        ast::Path {
                                            span: self.get_inner_span(&for_type),
                                            segments: Default::default(),
                                        },
                                        self.get_inner_span(&bound),
                                    ),
                                    bound_mod,
                                )
                            }),
                    }
                })
                .collect()
        } else {
            Default::default()
        }
    }

    fn to_generic_params(&self, node: Option<ra::TypeParamList>) -> Vec<ast::GenericParam> {
        if let Some(type_params) = node {
            type_params
                .generic_params()
                .map(|param| match param {
                    ra::GenericParam::TypeParam(type_param) => ast::GenericParam {
                        id: ast::DUMMY_NODE_ID,
                        ident: self.to_ident_from_name_or_err(
                            type_param.name(),
                            &type_param,
                            "type parameter",
                        ),
                        kind: ast::GenericParamKind::Type {
                            default: type_param
                                .default_type()
                                .map(|default_type| self.to_ty(default_type)),
                        },
                        attrs: self.get_attributes(&type_param).collect::<Vec<_>>().into(),
                        bounds: self.get_generic_bounds(&type_param),
                        is_placeholder: false,
                    },
                    ra::GenericParam::LifetimeParam(lifetime_param) => ast::GenericParam {
                        id: ast::DUMMY_NODE_ID,
                        ident: self
                            .to_lifetime_or_err(
                                lifetime_param.lifetime(),
                                &lifetime_param,
                                "lifetime parameter",
                            )
                            .ident,
                        kind: ast::GenericParamKind::Lifetime,
                        attrs: self.get_attributes(&lifetime_param).collect::<Vec<_>>().into(),
                        bounds: lifetime_param
                            .lifetime_bounds()
                            .map(|lifetime| ast::GenericBound::Outlives(self.to_lifetime(lifetime)))
                            .collect(),
                        is_placeholder: false,
                    },
                    ra::GenericParam::ConstParam(const_param) => ast::GenericParam {
                        id: ast::DUMMY_NODE_ID,
                        ident: self.to_ident_from_name_or_err(
                            const_param.name(),
                            &const_param,
                            "const parameter",
                        ),
                        kind: ast::GenericParamKind::Const {
                            ty: self.to_ty_or_err(
                                const_param.ascribed_type(),
                                &const_param,
                                "const parameter",
                            ),
                        },
                        attrs: self.get_attributes(&const_param).collect::<Vec<_>>().into(),
                        bounds: Default::default(),
                        is_placeholder: false,
                    },
                })
                .collect()
        } else {
            Default::default()
        }
    }

    fn to_symbol_from_text(&self, text: &SmolStr) -> Symbol {
        // TODO: ideally we should change the rustc interner to use SmolStr instead, to avoid double interning
        Symbol::intern(text.as_str())
    }

    fn to_symbol_from_token(&self, token: impl ra::AstToken) -> Symbol {
        self.to_symbol_from_text(&token.text())
    }

    fn to_ident_from_text(&self, range: TextRange, text: &SmolStr) -> ast::Ident {
        ast::Ident { span: self.to_span(range), name: self.to_symbol_from_text(text) }
    }

    fn to_ident_from_name_ref(&self, node: ra::NameRef) -> ast::Ident {
        self.to_ident_from_text(inner_range(&node), &node.text())
    }

    fn to_ident_from_name_ref_or_err(
        &self,
        name_ref_opt: Option<ra::NameRef>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> ast::Ident {
        self.expect_in(name_ref_opt, element, "name reference", container)
            .map(|name| self.to_ident_from_name_ref(name))
            .unwrap_or_else(|| ast::Ident::new(kw::Invalid, self.get_inner_span(element)))
    }

    fn to_ident_from_name(&self, node: ra::Name) -> ast::Ident {
        self.to_ident_from_text(inner_range(&node), &node.text())
    }

    fn to_ident_from_name_or_err(
        &self,
        name_opt: Option<ra::Name>,
        element: &impl ra::AstElement,
        container: &str,
    ) -> ast::Ident {
        self.expect_in(name_opt, element, "name", container)
            .map(|name| self.to_ident_from_name(name))
            .unwrap_or_else(|| ast::Ident::new(kw::Invalid, self.get_inner_span(element)))
    }

    fn to_ident_from_token(&self, token: impl ra::AstToken) -> ast::Ident {
        self.to_ident_from_text(inner_range(&token), &token.text())
    }

    fn to_tokens(&self, element: impl ra::AstElement) -> Option<TokenStream> {
        self.get_tokens_from_range(inner_range(&element))
    }

    fn get_tokens_from_range(&self, range: TextRange) -> Option<TokenStream> {
        let span = self.to_span(range);
        let mut srdr = lexer::StringReader::retokenize(&self.sess, span);
        srdr.pos = span.lo(); // somehow retokenize() doesn't set this: not sure how the other callers can possibly work properly, but we definitely need this
        let (token_trees, unmatched_braces) = srdr.into_token_trees();

        match token_trees {
            Ok(stream) => {
                //fix_token_stream(&mut stream);
                Some(stream)
            }
            Err(err) => {
                self.sess.span_diagnostic.emit_diagnostic(&err);
                for unmatched in unmatched_braces {
                    if let Some(err) = make_unclosed_delims_error(unmatched, &self.sess) {
                        self.sess.span_diagnostic.emit_diagnostic(&err);
                    }
                }
                None
            }
        }
    }

    fn get_collected_tokens_from_range(&self, range: TextRange) -> Option<TokenStream> {
        if let Some(mut token_stream) = self.get_tokens_from_range(range) {
            remove_joint(&mut token_stream);
            Some(token_stream)
        } else {
            None
        }
    }
}

fn make_unclosed_delims_error(
    unmatched: lexer::UnmatchedBrace,
    sess: &ParseSess,
) -> Option<DiagnosticBuilder<'_>> {
    // `None` here means an `Eof` was found. We already emit those errors elsewhere, we add them to
    // `unmatched_braces` only for error recovery in the `Parser`.
    let found_delim = unmatched.found_delim?;
    let mut err = sess.span_diagnostic.struct_span_err(
        unmatched.found_span,
        &format!(
            "mismatched closing delimiter: `{}`",
            rustc_ast_pretty::pprust::token_kind_to_string(&rustc_ast::token::CloseDelim(
                found_delim
            )),
        ),
    );
    err.span_label(unmatched.found_span, "mismatched closing delimiter");
    if let Some(sp) = unmatched.candidate_span {
        err.span_label(sp, "closing delimiter possibly meant for this");
    }
    if let Some(sp) = unmatched.unclosed_span {
        err.span_label(sp, "unclosed delimiter");
    }
    Some(err)
}
