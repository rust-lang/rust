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

use std::{iter, ops::Range, sync};

use base_db::SourceDatabase;
use expect_test::Expect;
use hir_expand::{
    db::ExpandDatabase,
    proc_macro::{ProcMacro, ProcMacroExpander, ProcMacroExpansionError, ProcMacroKind},
    span_map::SpanMapRef,
    InFile, MacroFileId, MacroFileIdExt,
};
use intern::Symbol;
use span::{Edition, Span};
use stdx::{format_to, format_to_acc};
use syntax::{
    ast::{self, edit::IndentLevel},
    AstNode,
    SyntaxKind::{COMMENT, EOF, IDENT, LIFETIME_IDENT},
    SyntaxNode, T,
};
use test_fixture::WithFixture;

use crate::{
    db::DefDatabase,
    nameres::{DefMap, MacroSubNs, ModuleSource},
    resolver::HasResolver,
    src::HasSource,
    test_db::TestDB,
    tt::Subtree,
    AdtId, AsMacroCall, Lookup, ModuleDefId,
};

#[track_caller]
fn check(ra_fixture: &str, mut expect: Expect) {
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
    let db = TestDB::with_files_extra_proc_macros(ra_fixture, extra_proc_macros);
    let krate = db.crate_graph().iter().next().unwrap();
    let def_map = db.crate_def_map(krate);
    let local_id = DefMap::ROOT;
    let module = def_map.module_id(local_id);
    let resolver = module.resolver(&db);
    let source = def_map[local_id].definition_source(&db);
    let source_file = match source.value {
        ModuleSource::SourceFile(it) => it,
        ModuleSource::Module(_) | ModuleSource::BlockExpr(_) => panic!(),
    };

    // What we want to do is to replace all macros (fn-like, derive, attr) with
    // their expansions. Turns out, we don't actually store enough information
    // to do this precisely though! Specifically, if a macro expands to nothing,
    // it leaves zero traces in def-map, so we can't get its expansion after the
    // fact.
    //
    // This is the usual
    // <https://github.com/rust-lang/rust-analyzer/issues/3407>
    // resolve/record tension!
    //
    // So here we try to do a resolve, which is necessary a heuristic. For macro
    // calls, we use `as_call_id_with_errors`. For derives, we look at the impls
    // in the module and assume that, if impls's source is a different
    // `HirFileId`, than it came from macro expansion.

    let mut text_edits = Vec::new();
    let mut expansions = Vec::new();

    for macro_call in source_file.syntax().descendants().filter_map(ast::MacroCall::cast) {
        let macro_call = InFile::new(source.file_id, &macro_call);
        let res = macro_call
            .as_call_id_with_errors(&db, krate, |path| {
                resolver
                    .resolve_path_as_macro(&db, path, Some(MacroSubNs::Bang))
                    .map(|(it, _)| db.macro_def(it))
            })
            .unwrap();
        let macro_call_id = res.value.unwrap();
        let macro_file = MacroFileId { macro_call_id };
        let mut expansion_result = db.parse_macro_expansion(macro_file);
        expansion_result.err = expansion_result.err.or(res.err);
        expansions.push((macro_call.value.clone(), expansion_result));
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
            format_to!(expn_text, "/* error: {} */", err.render_to_string(&db).0);
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

        if let Some(src) = src {
            if let Some(file_id) = src.file_id.macro_file() {
                if file_id.is_attr_macro(&db) || file_id.is_custom_derive(&db) {
                    let call = file_id.call_node(&db);
                    let mut show_spans = false;
                    let mut show_ctxt = false;
                    for comment in
                        call.value.children_with_tokens().filter(|it| it.kind() == COMMENT)
                    {
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
        }
    }

    for impl_id in def_map[local_id].scope.impls() {
        let src = impl_id.lookup(&db).source(&db);
        if let Some(macro_file) = src.file_id.macro_file() {
            if macro_file.is_builtin_derive(&db) {
                let pp = pretty_print_macro_expansion(
                    src.value.syntax().clone(),
                    db.span_map(macro_file.into()).as_ref(),
                    false,
                    false,
                );
                format_to!(expanded_text, "\n{}", pp)
            }
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
    for token in iter::successors(expn.first_token(), |t| t.next_token()) {
        let curr_kind = token.kind();
        let space = match (prev_kind, curr_kind) {
            _ if prev_kind.is_trivia() || curr_kind.is_trivia() => "",
            _ if prev_kind.is_literal() && !curr_kind.is_punct() => " ",
            (T!['{'], T!['}']) => "",
            (T![=], _) | (_, T![=]) => " ",
            (_, T!['{']) => " ",
            (T![;] | T!['{'] | T!['}'], _) => "\n",
            (_, T!['}']) => "\n",
            (IDENT | LIFETIME_IDENT, IDENT | LIFETIME_IDENT) => " ",
            _ if prev_kind.is_keyword(Edition::CURRENT)
                && curr_kind.is_keyword(Edition::CURRENT) =>
            {
                " "
            }
            (IDENT, _) if curr_kind.is_keyword(Edition::CURRENT) => " ",
            (_, IDENT) if prev_kind.is_keyword(Edition::CURRENT) => " ",
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
        subtree: &Subtree,
        _: Option<&Subtree>,
        _: &base_db::Env,
        _: Span,
        _: Span,
        _: Span,
        _: Option<String>,
    ) -> Result<Subtree, ProcMacroExpansionError> {
        let (parse, _) = syntax_bridge::token_tree_to_syntax_node(
            subtree,
            syntax_bridge::TopEntryPoint::MacroItems,
            span::Edition::CURRENT,
        );
        if parse.errors().is_empty() {
            Ok(subtree.clone())
        } else {
            panic!("got invalid macro input: {:?}", parse.errors());
        }
    }
}
