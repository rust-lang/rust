//! This module contains tests for macro expansion. Effectively, it covers `tt`,
//! `mbe`, `proc_macro_api` and `hir_expand` crates. This might seem like a
//! wrong architecture at the first glance, but is intentional.
//!
//! Physically, macro expansion process is intertwined with name resolution. You
//! can not expand *just* the syntax. So, to be able to write integration tests
//! of the "expand this code please" form, we have to do it after name
//! resolution. That is, in this crate. We *could* fake some dependencies and
//! write unit-tests (in fact, we used to do that), but that makes tests brittle
//! and harder to understand.

mod mbe;
mod builtin_fn_macro;
mod builtin_derive_macro;
mod proc_macros;

use std::{iter, ops::Range, sync};

use ::mbe::TokenMap;
use base_db::{fixture::WithFixture, ProcMacro, SourceDatabase};
use expect_test::Expect;
use hir_expand::{
    db::{ExpandDatabase, TokenExpander},
    AstId, InFile, MacroDefId, MacroDefKind, MacroFile,
};
use stdx::format_to;
use syntax::{
    ast::{self, edit::IndentLevel},
    AstNode, SyntaxElement,
    SyntaxKind::{self, COMMENT, EOF, IDENT, LIFETIME_IDENT},
    SyntaxNode, TextRange, T,
};
use tt::token_id::{Subtree, TokenId};

use crate::{
    db::DefDatabase,
    macro_id_to_def_id,
    nameres::{DefMap, MacroSubNs, ModuleSource},
    resolver::HasResolver,
    src::HasSource,
    test_db::TestDB,
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
            name: "identity_when_valid".into(),
            kind: base_db::ProcMacroKind::Attr,
            expander: sync::Arc::new(IdentityWhenValidProcMacroExpander),
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

    for macro_ in source_file.syntax().descendants().filter_map(ast::Macro::cast) {
        let mut show_token_ids = false;
        for comment in macro_.syntax().children_with_tokens().filter(|it| it.kind() == COMMENT) {
            show_token_ids |= comment.to_string().contains("+tokenids");
        }
        if !show_token_ids {
            continue;
        }

        let call_offset = macro_.syntax().text_range().start().into();
        let file_ast_id = db.ast_id_map(source.file_id).ast_id(&macro_);
        let ast_id = AstId::new(source.file_id, file_ast_id.upcast());
        let kind = MacroDefKind::Declarative(ast_id);

        let macro_def = db
            .macro_def(MacroDefId { krate, kind, local_inner: false, allow_internal_unsafe: false })
            .unwrap();
        if let TokenExpander::DeclarativeMacro { mac, def_site_token_map } = &*macro_def {
            let tt = match &macro_ {
                ast::Macro::MacroRules(mac) => mac.token_tree().unwrap(),
                ast::Macro::MacroDef(_) => unimplemented!(""),
            };

            let tt_start = tt.syntax().text_range().start();
            tt.syntax().descendants_with_tokens().filter_map(SyntaxElement::into_token).for_each(
                |token| {
                    let range = token.text_range().checked_sub(tt_start).unwrap();
                    if let Some(id) = def_site_token_map.token_by_range(range) {
                        let offset = (range.end() + tt_start).into();
                        text_edits.push((offset..offset, format!("#{}", id.0)));
                    }
                },
            );
            text_edits.push((
                call_offset..call_offset,
                format!("// call ids will be shifted by {:?}\n", mac.shift()),
            ));
        }
    }

    for macro_call in source_file.syntax().descendants().filter_map(ast::MacroCall::cast) {
        let macro_call = InFile::new(source.file_id, &macro_call);
        let res = macro_call
            .as_call_id_with_errors(&db, krate, |path| {
                resolver
                    .resolve_path_as_macro(&db, &path, Some(MacroSubNs::Bang))
                    .map(|it| macro_id_to_def_id(&db, it))
            })
            .unwrap();
        let macro_call_id = res.value.unwrap();
        let macro_file = MacroFile { macro_call_id };
        let mut expansion_result = db.parse_macro_expansion(macro_file);
        expansion_result.err = expansion_result.err.or(res.err);
        expansions.push((macro_call.value.clone(), expansion_result, db.macro_arg(macro_call_id)));
    }

    for (call, exp, arg) in expansions.into_iter().rev() {
        let mut tree = false;
        let mut expect_errors = false;
        let mut show_token_ids = false;
        for comment in call.syntax().children_with_tokens().filter(|it| it.kind() == COMMENT) {
            tree |= comment.to_string().contains("+tree");
            expect_errors |= comment.to_string().contains("+errors");
            show_token_ids |= comment.to_string().contains("+tokenids");
        }

        let mut expn_text = String::new();
        if let Some(err) = exp.err {
            format_to!(expn_text, "/* error: {} */", err);
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
                "parse errors in expansion: \n{:#?}",
                parse.errors()
            );
        }
        let pp = pretty_print_macro_expansion(
            parse.syntax_node(),
            show_token_ids.then_some(&*token_map),
        );
        let indent = IndentLevel::from_node(call.syntax());
        let pp = reindent(indent, pp);
        format_to!(expn_text, "{}", pp);

        if tree {
            let tree = format!("{:#?}", parse.syntax_node())
                .split_inclusive('\n')
                .map(|line| format!("// {line}"))
                .collect::<String>();
            format_to!(expn_text, "\n{}", tree)
        }
        let range = call.syntax().text_range();
        let range: Range<usize> = range.into();

        if show_token_ids {
            if let Some((tree, map, _)) = arg.as_deref() {
                let tt_range = call.token_tree().unwrap().syntax().text_range();
                let mut ranges = Vec::new();
                extract_id_ranges(&mut ranges, map, tree);
                for (range, id) in ranges {
                    let idx = (tt_range.start() + range.end()).into();
                    text_edits.push((idx..idx, format!("#{}", id.0)));
                }
            }
            text_edits.push((range.start..range.start, "// ".into()));
            call.to_string().match_indices('\n').for_each(|(offset, _)| {
                let offset = offset + 1 + range.start;
                text_edits.push((offset..offset, "// ".into()));
            });
            text_edits.push((range.end..range.end, "\n".into()));
            text_edits.push((range.end..range.end, expn_text));
        } else {
            text_edits.push((range, expn_text));
        }
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
            if src.file_id.is_attr_macro(&db) || src.file_id.is_custom_derive(&db) {
                let pp = pretty_print_macro_expansion(src.value, None);
                format_to!(expanded_text, "\n{}", pp)
            }
        }
    }

    for impl_id in def_map[local_id].scope.impls() {
        let src = impl_id.lookup(&db).source(&db);
        if src.file_id.is_builtin_derive(&db).is_some() {
            let pp = pretty_print_macro_expansion(src.value.syntax().clone(), None);
            format_to!(expanded_text, "\n{}", pp)
        }
    }

    expect.indent(false);
    expect.assert_eq(&expanded_text);
}

fn extract_id_ranges(ranges: &mut Vec<(TextRange, TokenId)>, map: &TokenMap, tree: &Subtree) {
    tree.token_trees.iter().for_each(|tree| match tree {
        tt::TokenTree::Leaf(leaf) => {
            let id = match leaf {
                tt::Leaf::Literal(it) => it.span,
                tt::Leaf::Punct(it) => it.span,
                tt::Leaf::Ident(it) => it.span,
            };
            ranges.extend(map.ranges_by_token(id, SyntaxKind::ERROR).map(|range| (range, id)));
        }
        tt::TokenTree::Subtree(tree) => extract_id_ranges(ranges, map, tree),
    });
}

fn reindent(indent: IndentLevel, pp: String) -> String {
    if !pp.contains('\n') {
        return pp;
    }
    let mut lines = pp.split_inclusive('\n');
    let mut res = lines.next().unwrap().to_string();
    for line in lines {
        if line.trim().is_empty() {
            res.push_str(line)
        } else {
            format_to!(res, "{}{}", indent, line)
        }
    }
    res
}

fn pretty_print_macro_expansion(expn: SyntaxNode, map: Option<&TokenMap>) -> String {
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
            _ if prev_kind.is_keyword() && curr_kind.is_keyword() => " ",
            (IDENT, _) if curr_kind.is_keyword() => " ",
            (_, IDENT) if prev_kind.is_keyword() => " ",
            (T![>], IDENT) => " ",
            (T![>], _) if curr_kind.is_keyword() => " ",
            (T![->], _) | (_, T![->]) => " ",
            (T![&&], _) | (_, T![&&]) => " ",
            (T![,], _) => " ",
            (T![:], IDENT | T!['(']) => " ",
            (T![:], _) if curr_kind.is_keyword() => " ",
            (T![fn], T!['(']) => "",
            (T![']'], _) if curr_kind.is_keyword() => " ",
            (T![']'], T![#]) => "\n",
            (T![Self], T![::]) => "",
            _ if prev_kind.is_keyword() => " ",
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
        if let Some(map) = map {
            if let Some(id) = map.token_by_range(token.text_range()) {
                format_to!(res, "#{}", id.0);
            }
        }
    }
    res
}

// Identity mapping, but only works when the input is syntactically valid. This
// simulates common proc macros that unnecessarily parse their input and return
// compile errors.
#[derive(Debug)]
struct IdentityWhenValidProcMacroExpander;
impl base_db::ProcMacroExpander for IdentityWhenValidProcMacroExpander {
    fn expand(
        &self,
        subtree: &Subtree,
        _: Option<&Subtree>,
        _: &base_db::Env,
    ) -> Result<Subtree, base_db::ProcMacroExpansionError> {
        let (parse, _) =
            ::mbe::token_tree_to_syntax_node(subtree, ::mbe::TopEntryPoint::MacroItems);
        if parse.errors().is_empty() {
            Ok(subtree.clone())
        } else {
            panic!("got invalid macro input: {:?}", parse.errors());
        }
    }
}
