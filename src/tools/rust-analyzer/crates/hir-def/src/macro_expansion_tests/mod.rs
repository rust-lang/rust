//! This module contains integration tests for macro expansion with name resolution. Effectively, it
//! covers `tt`, `mbe`, `proc_macro_api` and `hir_expand` crates. This might seem like a  wrong
//! architecture at the first glance, but is intentional.
//!
//! Physically, macro expansion process is intertwined with name resolution. You
//! can not expand *just* the syntax. So, to be able to write integration tests
//! of the "expand this code please" form, we have to do it after name
//! resolution. That is, in this crate. We *could* fake some dependencies and
//! write unit-tests (in fact, we used to do that), but that makes tests brittle
//! and harder to understand.

mod builtin_derive_macro;
mod builtin_fn_macro;
mod mbe;
mod proc_macros;

use std::{any::TypeId, iter, ops::Range, sync};

use base_db::RootQueryDb;
use expect_test::Expect;
use hir_expand::{
    AstId, InFile, MacroCallId, MacroCallKind, MacroKind,
    builtin::quote::quote,
    db::ExpandDatabase,
    proc_macro::{ProcMacro, ProcMacroExpander, ProcMacroExpansionError, ProcMacroKind},
    span_map::SpanMapRef,
};
use intern::{Symbol, sym};
use itertools::Itertools;
use span::{Edition, ROOT_ERASED_FILE_AST_ID, Span, SpanAnchor, SyntaxContext};
use stdx::{format_to, format_to_acc};
use syntax::{
    AstNode, AstPtr,
    SyntaxKind::{COMMENT, EOF, IDENT, LIFETIME_IDENT},
    SyntaxNode, T,
    ast::{self, edit::IndentLevel},
};
use syntax_bridge::token_tree_to_syntax_node;
use test_fixture::WithFixture;
use tt::{TextRange, TextSize};

use crate::{
    AdtId, Lookup, ModuleDefId,
    db::DefDatabase,
    nameres::{DefMap, ModuleSource, crate_def_map},
    src::HasSource,
    test_db::TestDB,
    tt::TopSubtree,
};

#[track_caller]
fn check_errors(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let db = TestDB::with_files(ra_fixture);
    let krate = db.fetch_test_crate();
    let def_map = crate_def_map(&db, krate);
    let errors = def_map
        .modules()
        .flat_map(|module| module.1.scope.all_macro_calls())
        .filter_map(|macro_call| {
            let errors = db.parse_macro_expansion_error(macro_call)?;
            let errors = errors.err.as_ref()?.render_to_string(&db);
            let macro_loc = db.lookup_intern_macro_call(macro_call);
            let ast_id = match macro_loc.kind {
                MacroCallKind::FnLike { ast_id, .. } => ast_id.map(|it| it.erase()),
                MacroCallKind::Derive { ast_id, .. } => ast_id.map(|it| it.erase()),
                MacroCallKind::Attr { ast_id, .. } => ast_id.map(|it| it.erase()),
            };

            let editioned_file_id =
                ast_id.file_id.file_id().expect("macros inside macros are not supported");

            let ast = db.parse(editioned_file_id).syntax_node();
            let ast_id_map = db.ast_id_map(ast_id.file_id);
            let node = ast_id_map.get_erased(ast_id.value).to_node(&ast);
            Some((node.text_range(), errors))
        })
        .sorted_unstable_by_key(|(range, _)| range.start())
        .format_with("\n", |(range, err), format| format(&format_args!("{range:?}: {err}")))
        .to_string();
    expect.assert_eq(&errors);
}

fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, mut expect: Expect) {
    let extra_proc_macros = vec![(
        r#"
#[proc_macro_attribute]
pub fn identity_when_valid(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
"#
        .into(),
        ProcMacro {
            name: Symbol::intern("identity_when_valid"),
            kind: ProcMacroKind::Attr,
            expander: sync::Arc::new(IdentityWhenValidProcMacroExpander),
            disabled: false,
        },
    )];

    fn resolve(
        db: &dyn DefDatabase,
        def_map: &DefMap,
        ast_id: AstId<ast::MacroCall>,
        ast_ptr: InFile<AstPtr<ast::MacroCall>>,
    ) -> Option<MacroCallId> {
        def_map.modules().find_map(|module| {
            for decl in
                module.1.scope.declarations().chain(module.1.scope.unnamed_consts().map(Into::into))
            {
                let body = match decl {
                    ModuleDefId::FunctionId(it) => it.into(),
                    ModuleDefId::ConstId(it) => it.into(),
                    ModuleDefId::StaticId(it) => it.into(),
                    _ => continue,
                };

                let (body, sm) = db.body_with_source_map(body);
                if let Some(it) =
                    body.blocks(db).find_map(|block| resolve(db, block.1, ast_id, ast_ptr))
                {
                    return Some(it);
                }
                if let Some((_, res)) = sm.macro_calls().find(|it| it.0 == ast_ptr) {
                    return Some(res);
                }
            }
            module.1.scope.macro_invoc(ast_id)
        })
    }

    let db = TestDB::with_files_extra_proc_macros(ra_fixture, extra_proc_macros);
    let krate = db.fetch_test_crate();
    let def_map = crate_def_map(&db, krate);
    let local_id = DefMap::ROOT;
    let source = def_map[local_id].definition_source(&db);
    let source_file = match source.value {
        ModuleSource::SourceFile(it) => it,
        ModuleSource::Module(_) | ModuleSource::BlockExpr(_) => panic!(),
    };

    let mut text_edits = Vec::new();
    let mut expansions = Vec::new();

    for macro_call_node in source_file.syntax().descendants().filter_map(ast::MacroCall::cast) {
        let ast_id = db.ast_id_map(source.file_id).ast_id(&macro_call_node);
        let ast_id = InFile::new(source.file_id, ast_id);
        let ptr = InFile::new(source.file_id, AstPtr::new(&macro_call_node));
        let macro_call_id = resolve(&db, def_map, ast_id, ptr)
            .unwrap_or_else(|| panic!("unable to find semantic macro call {macro_call_node}"));
        let expansion_result = db.parse_macro_expansion(macro_call_id);
        expansions.push((macro_call_node.clone(), expansion_result));
    }

    for (call, exp) in expansions.into_iter().rev() {
        let mut tree = false;
        let mut expect_errors = false;
        let mut show_spans = false;
        let mut show_ctxt = false;
        for comment in call.syntax().children_with_tokens().filter(|it| it.kind() == COMMENT) {
            tree |= comment.to_string().contains("+tree");
            expect_errors |= comment.to_string().contains("+errors");
            show_spans |= comment.to_string().contains("+spans");
            show_ctxt |= comment.to_string().contains("+syntaxctxt");
        }

        let mut expn_text = String::new();
        if let Some(err) = exp.err {
            format_to!(expn_text, "/* error: {} */", err.render_to_string(&db).message);
        }
        let (parse, token_map) = exp.value;
        if expect_errors {
            assert!(!parse.errors().is_empty(), "no parse errors in expansion");
            for e in parse.errors() {
                format_to!(expn_text, "/* parse error: {} */\n", e);
            }
        } else {
            assert!(
                parse.errors().is_empty(),
                "parse errors in expansion: \n{:#?}\n```\n{}\n```",
                parse.errors(),
                parse.syntax_node(),
            );
        }
        let pp = pretty_print_macro_expansion(
            parse.syntax_node(),
            SpanMapRef::ExpansionSpanMap(&token_map),
            show_spans,
            show_ctxt,
        );
        let indent = IndentLevel::from_node(call.syntax());
        let pp = reindent(indent, pp);
        format_to!(expn_text, "{}", pp);

        if tree {
            let tree = format!("{:#?}", parse.syntax_node())
                .split_inclusive('\n')
                .fold(String::new(), |mut acc, line| format_to_acc!(acc, "// {line}"));
            format_to!(expn_text, "\n{}", tree)
        }
        let range = call.syntax().text_range();
        let range: Range<usize> = range.into();
        text_edits.push((range, expn_text));
    }

    text_edits.sort_by_key(|(range, _)| range.start);
    text_edits.reverse();
    let mut expanded_text = source_file.to_string();
    for (range, text) in text_edits {
        expanded_text.replace_range(range, &text);
    }

    for decl_id in def_map[local_id].scope.declarations() {
        // FIXME: I'm sure there's already better way to do this
        let src = match decl_id {
            ModuleDefId::AdtId(AdtId::StructId(struct_id)) => {
                Some(struct_id.lookup(&db).source(&db).syntax().cloned())
            }
            ModuleDefId::FunctionId(function_id) => {
                Some(function_id.lookup(&db).source(&db).syntax().cloned())
            }
            _ => None,
        };

        if let Some(src) = src
            && let Some(file_id) = src.file_id.macro_file()
            && let MacroKind::Derive
            | MacroKind::DeriveBuiltIn
            | MacroKind::Attr
            | MacroKind::AttrBuiltIn = file_id.kind(&db)
        {
            let call = file_id.call_node(&db);
            let mut show_spans = false;
            let mut show_ctxt = false;
            for comment in call.value.children_with_tokens().filter(|it| it.kind() == COMMENT) {
                show_spans |= comment.to_string().contains("+spans");
                show_ctxt |= comment.to_string().contains("+syntaxctxt");
            }
            let pp = pretty_print_macro_expansion(
                src.value,
                db.span_map(src.file_id).as_ref(),
                show_spans,
                show_ctxt,
            );
            format_to!(expanded_text, "\n{}", pp)
        }
    }

    for impl_id in def_map[local_id].scope.impls() {
        let src = impl_id.lookup(&db).source(&db);
        if let Some(macro_file) = src.file_id.macro_file()
            && let MacroKind::DeriveBuiltIn | MacroKind::Derive = macro_file.kind(&db)
        {
            let pp = pretty_print_macro_expansion(
                src.value.syntax().clone(),
                db.span_map(macro_file.into()).as_ref(),
                false,
                false,
            );
            format_to!(expanded_text, "\n{}", pp)
        }
    }

    expect.indent(false);
    expect.assert_eq(&expanded_text);
}

fn reindent(indent: IndentLevel, pp: String) -> String {
    if !pp.contains('\n') {
        return pp;
    }
    let mut lines = pp.split_inclusive('\n');
    let mut res = lines.next().unwrap().to_owned();
    for line in lines {
        if line.trim().is_empty() {
            res.push_str(line)
        } else {
            format_to!(res, "{}{}", indent, line)
        }
    }
    res
}

fn pretty_print_macro_expansion(
    expn: SyntaxNode,
    map: SpanMapRef<'_>,
    show_spans: bool,
    show_ctxt: bool,
) -> String {
    let mut res = String::new();
    let mut prev_kind = EOF;
    let mut indent_level = 0;
    for token in iter::successors(expn.first_token(), |t| t.next_token())
        .take_while(|token| token.text_range().start() < expn.text_range().end())
    {
        let curr_kind = token.kind();
        let space = match (prev_kind, curr_kind) {
            _ if prev_kind.is_trivia() || curr_kind.is_trivia() => "",
            _ if prev_kind.is_literal() && !curr_kind.is_punct() => " ",
            (T!['{'], T!['}']) => "",
            (T![=], _) | (_, T![=]) => " ",
            (_, T!['{']) => " ",
            (T![;] | T!['{'] | T!['}'], _) => "\n",
            (_, T!['}']) => "\n",
            _ if (prev_kind.is_any_identifier()
                || prev_kind == LIFETIME_IDENT
                || prev_kind.is_literal())
                && (curr_kind.is_any_identifier()
                    || curr_kind == LIFETIME_IDENT
                    || curr_kind.is_literal()) =>
            {
                " "
            }
            (T![>], IDENT) => " ",
            (T![>], _) if curr_kind.is_keyword(Edition::CURRENT) => " ",
            (T![->], _) | (_, T![->]) => " ",
            (T![&&], _) | (_, T![&&]) => " ",
            (T![,], _) => " ",
            (T![:], IDENT | T!['(']) => " ",
            (T![:], _) if curr_kind.is_keyword(Edition::CURRENT) => " ",
            (T![fn], T!['(']) => "",
            (T![']'], _) if curr_kind.is_keyword(Edition::CURRENT) => " ",
            (T![']'], T![#]) => "\n",
            (T![Self], T![::]) => "",
            _ if prev_kind.is_keyword(Edition::CURRENT) => " ",
            _ => "",
        };

        match prev_kind {
            T!['{'] => indent_level += 1,
            T!['}'] => indent_level -= 1,
            _ => (),
        }

        res.push_str(space);
        if space == "\n" {
            let level = if curr_kind == T!['}'] { indent_level - 1 } else { indent_level };
            res.push_str(&"    ".repeat(level));
        }
        prev_kind = curr_kind;
        format_to!(res, "{}", token);
        if show_spans || show_ctxt {
            let span = map.span_for_range(token.text_range());
            format_to!(res, "#");
            if show_spans {
                format_to!(res, "{span}",);
            } else if show_ctxt {
                format_to!(res, "\\{}", span.ctx);
            }
            format_to!(res, "#");
        }
    }
    res
}

// Identity mapping, but only works when the input is syntactically valid. This
// simulates common proc macros that unnecessarily parse their input and return
// compile errors.
#[derive(Debug)]
struct IdentityWhenValidProcMacroExpander;
impl ProcMacroExpander for IdentityWhenValidProcMacroExpander {
    fn expand(
        &self,
        subtree: &TopSubtree,
        _: Option<&TopSubtree>,
        _: &base_db::Env,
        _: Span,
        _: Span,
        _: Span,
        _: String,
    ) -> Result<TopSubtree, ProcMacroExpansionError> {
        let (parse, _) = syntax_bridge::token_tree_to_syntax_node(
            subtree,
            syntax_bridge::TopEntryPoint::MacroItems,
            &mut |_| span::Edition::CURRENT,
            span::Edition::CURRENT,
        );
        if parse.errors().is_empty() {
            Ok(subtree.clone())
        } else {
            panic!("got invalid macro input: {:?}", parse.errors());
        }
    }

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool {
        other.type_id() == TypeId::of::<Self>()
    }
}

#[test]
fn regression_20171() {
    // This really isn't the appropriate place to put this test, but it's convenient with access to `quote!`.
    let span = Span {
        range: TextRange::empty(TextSize::new(0)),
        anchor: SpanAnchor {
            file_id: span::EditionedFileId::current_edition(span::FileId::from_raw(0)),
            ast_id: ROOT_ERASED_FILE_AST_ID,
        },
        ctx: SyntaxContext::root(Edition::CURRENT),
    };
    let close_brace = tt::Punct { char: '}', spacing: tt::Spacing::Alone, span };
    let dotdot1 = tt::Punct { char: '.', spacing: tt::Spacing::Joint, span };
    let dotdot2 = tt::Punct { char: '.', spacing: tt::Spacing::Alone, span };
    let dollar_crate = sym::dollar_crate;
    let tt = quote! {
            span => {
        if !((matches!(
            drive_parser(&mut parser, data, false),
            Err(TarParserError::CorruptField {
                field: CorruptFieldContext::PaxKvLength,
                error: GeneralParseError::ParseInt(ParseIntError { #dotdot1 #dotdot2 })
            })
        #close_brace  ))) {
        #dollar_crate::panic::panic_2021!();
    }}
        };
    token_tree_to_syntax_node(
        &tt,
        syntax_bridge::TopEntryPoint::MacroStmts,
        &mut |_| Edition::CURRENT,
        Edition::CURRENT,
    );
}
