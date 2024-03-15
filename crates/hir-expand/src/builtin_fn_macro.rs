//! Builtin macro

use base_db::{AnchoredPath, Edition, FileId};
use cfg::CfgExpr;
use either::Either;
use itertools::Itertools;
use mbe::{parse_exprs_with_sep, parse_to_token_tree};
use span::{Span, SpanAnchor, SyntaxContextId, ROOT_ERASED_FILE_AST_ID};
use syntax::ast::{self, AstToken};

use crate::{
    db::ExpandDatabase,
    hygiene::{span_with_call_site_ctxt, span_with_def_site_ctxt},
    name::{self, known},
    quote,
    quote::dollar_crate,
    tt::{self, DelimSpan},
    ExpandError, ExpandResult, HirFileIdExt, MacroCallId, MacroFileIdExt,
};

macro_rules! register_builtin {
    ( $LAZY:ident: $(($name:ident, $kind: ident) => $expand:ident),* , $EAGER:ident: $(($e_name:ident, $e_kind: ident) => $e_expand:ident),*  ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $LAZY {
            $($kind),*
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $EAGER {
            $($e_kind),*
        }

        impl BuiltinFnLikeExpander {
            pub fn expander(&self) -> fn (&dyn ExpandDatabase, MacroCallId, &tt::Subtree, Span) -> ExpandResult<tt::Subtree>  {
                match *self {
                    $( BuiltinFnLikeExpander::$kind => $expand, )*
                }
            }
        }

        impl EagerExpander {
            pub fn expander(&self) -> fn (&dyn ExpandDatabase, MacroCallId, &tt::Subtree, Span) -> ExpandResult<tt::Subtree>  {
                match *self {
                    $( EagerExpander::$e_kind => $e_expand, )*
                }
            }
        }

        fn find_by_name(ident: &name::Name) -> Option<Either<BuiltinFnLikeExpander, EagerExpander>> {
            match ident {
                $( id if id == &name::name![$name] => Some(Either::Left(BuiltinFnLikeExpander::$kind)), )*
                $( id if id == &name::name![$e_name] => Some(Either::Right(EagerExpander::$e_kind)), )*
                _ => return None,
            }
        }
    };
}

impl BuiltinFnLikeExpander {
    pub fn expand(
        &self,
        db: &dyn ExpandDatabase,
        id: MacroCallId,
        tt: &tt::Subtree,
        span: Span,
    ) -> ExpandResult<tt::Subtree> {
        let span = span_with_def_site_ctxt(db, span, id);
        self.expander()(db, id, tt, span)
    }
}

impl EagerExpander {
    pub fn expand(
        &self,
        db: &dyn ExpandDatabase,
        id: MacroCallId,
        tt: &tt::Subtree,
        span: Span,
    ) -> ExpandResult<tt::Subtree> {
        let span = span_with_def_site_ctxt(db, span, id);
        self.expander()(db, id, tt, span)
    }

    pub fn is_include(&self) -> bool {
        matches!(self, EagerExpander::Include)
    }

    pub fn is_include_like(&self) -> bool {
        matches!(
            self,
            EagerExpander::Include | EagerExpander::IncludeStr | EagerExpander::IncludeBytes
        )
    }

    pub fn is_env_or_option_env(&self) -> bool {
        matches!(self, EagerExpander::Env | EagerExpander::OptionEnv)
    }
}

pub fn find_builtin_macro(
    ident: &name::Name,
) -> Option<Either<BuiltinFnLikeExpander, EagerExpander>> {
    find_by_name(ident)
}

register_builtin! {
    BuiltinFnLikeExpander:
    (column, Column) => line_expand,
    (file, File) => file_expand,
    (line, Line) => line_expand,
    (module_path, ModulePath) => module_path_expand,
    (assert, Assert) => assert_expand,
    (stringify, Stringify) => stringify_expand,
    (llvm_asm, LlvmAsm) => asm_expand,
    (asm, Asm) => asm_expand,
    (global_asm, GlobalAsm) => global_asm_expand,
    (cfg, Cfg) => cfg_expand,
    (core_panic, CorePanic) => panic_expand,
    (std_panic, StdPanic) => panic_expand,
    (unreachable, Unreachable) => unreachable_expand,
    (log_syntax, LogSyntax) => log_syntax_expand,
    (trace_macros, TraceMacros) => trace_macros_expand,
    (format_args, FormatArgs) => format_args_expand,
    (const_format_args, ConstFormatArgs) => format_args_expand,
    (format_args_nl, FormatArgsNl) => format_args_nl_expand,
    (quote, Quote) => quote_expand,

    EagerExpander:
    (compile_error, CompileError) => compile_error_expand,
    (concat, Concat) => concat_expand,
    (concat_idents, ConcatIdents) => concat_idents_expand,
    (concat_bytes, ConcatBytes) => concat_bytes_expand,
    (include, Include) => include_expand,
    (include_bytes, IncludeBytes) => include_bytes_expand,
    (include_str, IncludeStr) => include_str_expand,
    (env, Env) => env_expand,
    (option_env, OptionEnv) => option_env_expand
}

fn mk_pound(span: Span) -> tt::Subtree {
    crate::quote::IntoTt::to_subtree(
        vec![crate::tt::Leaf::Punct(crate::tt::Punct {
            char: '#',
            spacing: crate::tt::Spacing::Alone,
            span,
        })
        .into()],
        span,
    )
}

fn module_path_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    // Just return a dummy result.
    ExpandResult::ok(quote! {span =>
         "module::path"
    })
}

fn line_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    // dummy implementation for type-checking purposes
    // Note that `line!` and `column!` will never be implemented properly, as they are by definition
    // not incremental
    ExpandResult::ok(tt::Subtree {
        delimiter: tt::Delimiter::invisible_spanned(span),
        token_trees: Box::new([tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
            text: "0u32".into(),
            span,
        }))]),
    })
}

fn log_syntax_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    ExpandResult::ok(quote! {span =>})
}

fn trace_macros_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    ExpandResult::ok(quote! {span =>})
}

fn stringify_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let pretty = ::tt::pretty(&tt.token_trees);

    let expanded = quote! {span =>
        #pretty
    };

    ExpandResult::ok(expanded)
}

fn assert_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let call_site_span = span_with_call_site_ctxt(db, span, id);
    let args = parse_exprs_with_sep(tt, ',', call_site_span);
    let dollar_crate = dollar_crate(span);
    let expanded = match &*args {
        [cond, panic_args @ ..] => {
            let comma = tt::Subtree {
                delimiter: tt::Delimiter::invisible_spanned(call_site_span),
                token_trees: Box::new([tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct {
                    char: ',',
                    spacing: tt::Spacing::Alone,
                    span: call_site_span,
                }))]),
            };
            let cond = cond.clone();
            let panic_args = itertools::Itertools::intersperse(panic_args.iter().cloned(), comma);
            let mac = if use_panic_2021(db, span) {
                quote! {call_site_span => #dollar_crate::panic::panic_2021!(##panic_args) }
            } else {
                quote! {call_site_span => #dollar_crate::panic!(##panic_args) }
            };
            quote! {call_site_span =>{
                if !(#cond) {
                    #mac;
                }
            }}
        }
        [] => quote! {call_site_span =>{}},
    };

    ExpandResult::ok(expanded)
}

fn file_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    // FIXME: RA purposefully lacks knowledge of absolute file names
    // so just return "".
    let file_name = "";

    let expanded = quote! {span =>
        #file_name
    };

    ExpandResult::ok(expanded)
}

fn format_args_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    format_args_expand_general(db, id, tt, "", span)
}

fn format_args_nl_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    format_args_expand_general(db, id, tt, "\\n", span)
}

fn format_args_expand_general(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
    // FIXME: Make use of this so that mir interpretation works properly
    _end_string: &str,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let pound = mk_pound(span);
    let mut tt = tt.clone();
    tt.delimiter.kind = tt::DelimiterKind::Parenthesis;
    ExpandResult::ok(quote! {span =>
        builtin #pound format_args #tt
    })
}

fn asm_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    // We expand all assembly snippets to `format_args!` invocations to get format syntax
    // highlighting for them.
    let mut literals = Vec::new();
    for tt in tt.token_trees.chunks(2) {
        match tt {
            [tt::TokenTree::Leaf(tt::Leaf::Literal(lit))]
            | [tt::TokenTree::Leaf(tt::Leaf::Literal(lit)), tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: ',', span: _, spacing: _ }))] =>
            {
                let dollar_krate = dollar_crate(span);
                literals.push(quote!(span=>#dollar_krate::format_args!(#lit);));
            }
            _ => break,
        }
    }

    let pound = mk_pound(span);
    let expanded = quote! {span =>
        builtin #pound asm (
            {##literals}
        )
    };
    ExpandResult::ok(expanded)
}

fn global_asm_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    // Expand to nothing (at item-level)
    ExpandResult::ok(quote! {span =>})
}

fn cfg_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let loc = db.lookup_intern_macro_call(id);
    let expr = CfgExpr::parse(tt);
    let enabled = db.crate_graph()[loc.krate].cfg_options.check(&expr) != Some(false);
    let expanded = if enabled { quote!(span=>true) } else { quote!(span=>false) };
    ExpandResult::ok(expanded)
}

fn panic_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let dollar_crate = dollar_crate(span);
    let call_site_span = span_with_call_site_ctxt(db, span, id);

    let mac =
        if use_panic_2021(db, call_site_span) { known::panic_2021 } else { known::panic_2015 };

    // Expand to a macro call `$crate::panic::panic_{edition}`
    let mut call = quote!(call_site_span =>#dollar_crate::panic::#mac!);

    // Pass the original arguments
    let mut subtree = tt.clone();
    subtree.delimiter = tt::Delimiter {
        open: call_site_span,
        close: call_site_span,
        kind: tt::DelimiterKind::Parenthesis,
    };

    // FIXME(slow): quote! have a way to expand to builder to make this a vec!
    call.push(tt::TokenTree::Subtree(subtree));

    ExpandResult::ok(call)
}

fn unreachable_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let dollar_crate = dollar_crate(span);
    let call_site_span = span_with_call_site_ctxt(db, span, id);

    let mac = if use_panic_2021(db, call_site_span) {
        known::unreachable_2021
    } else {
        known::unreachable_2015
    };

    // Expand to a macro call `$crate::panic::panic_{edition}`
    let mut call = quote!(call_site_span =>#dollar_crate::panic::#mac!);

    // Pass the original arguments
    let mut subtree = tt.clone();
    subtree.delimiter = tt::Delimiter {
        open: call_site_span,
        close: call_site_span,
        kind: tt::DelimiterKind::Parenthesis,
    };

    // FIXME(slow): quote! have a way to expand to builder to make this a vec!
    call.push(tt::TokenTree::Subtree(subtree));

    ExpandResult::ok(call)
}

#[allow(clippy::never_loop)]
fn use_panic_2021(db: &dyn ExpandDatabase, span: Span) -> bool {
    // To determine the edition, we check the first span up the expansion
    // stack that does not have #[allow_internal_unstable(edition_panic)].
    // (To avoid using the edition of e.g. the assert!() or debug_assert!() definition.)
    loop {
        let Some(expn) = db.lookup_intern_syntax_context(span.ctx).outer_expn else {
            break false;
        };
        let expn = db.lookup_intern_macro_call(expn);
        // FIXME: Record allow_internal_unstable in the macro def (not been done yet because it
        // would consume quite a bit extra memory for all call locs...)
        // if let Some(features) = expn.def.allow_internal_unstable {
        //     if features.iter().any(|&f| f == sym::edition_panic) {
        //         span = expn.call_site;
        //         continue;
        //     }
        // }
        break expn.def.edition >= Edition::Edition2021;
    }
}

fn unquote_str(lit: &tt::Literal) -> Option<(String, Span)> {
    let span = lit.span;
    let lit = ast::make::tokens::literal(&lit.to_string());
    let token = ast::String::cast(lit)?;
    token.value().map(|it| (it.into_owned(), span))
}

fn unquote_char(lit: &tt::Literal) -> Option<(char, Span)> {
    let span = lit.span;
    let lit = ast::make::tokens::literal(&lit.to_string());
    let token = ast::Char::cast(lit)?;
    token.value().zip(Some(span))
}

fn unquote_byte_string(lit: &tt::Literal) -> Option<(Vec<u8>, Span)> {
    let span = lit.span;
    let lit = ast::make::tokens::literal(&lit.to_string());
    let token = ast::ByteString::cast(lit)?;
    token.value().map(|it| (it.into_owned(), span))
}

fn compile_error_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let err = match &*tt.token_trees {
        [tt::TokenTree::Leaf(tt::Leaf::Literal(it))] => match unquote_str(it) {
            Some((unquoted, _)) => ExpandError::other(unquoted.into_boxed_str()),
            None => ExpandError::other("`compile_error!` argument must be a string"),
        },
        _ => ExpandError::other("`compile_error!` argument must be a string"),
    };

    ExpandResult { value: quote! {span =>}, err: Some(err) }
}

fn concat_expand(
    _db: &dyn ExpandDatabase,
    _arg_id: MacroCallId,
    tt: &tt::Subtree,
    _: Span,
) -> ExpandResult<tt::Subtree> {
    let mut err = None;
    let mut text = String::new();
    let mut span: Option<Span> = None;
    let mut record_span = |s: Span| match &mut span {
        Some(span) if span.anchor == s.anchor => span.range = span.range.cover(s.range),
        Some(_) => (),
        None => span = Some(s),
    };
    for (i, mut t) in tt.token_trees.iter().enumerate() {
        // FIXME: hack on top of a hack: `$e:expr` captures get surrounded in parentheses
        // to ensure the right parsing order, so skip the parentheses here. Ideally we'd
        // implement rustc's model. cc https://github.com/rust-lang/rust-analyzer/pull/10623
        if let tt::TokenTree::Subtree(tt::Subtree { delimiter: delim, token_trees }) = t {
            if let [tt] = &**token_trees {
                if delim.kind == tt::DelimiterKind::Parenthesis {
                    t = tt;
                }
            }
        }

        match t {
            tt::TokenTree::Leaf(tt::Leaf::Literal(it)) if i % 2 == 0 => {
                // concat works with string and char literals, so remove any quotes.
                // It also works with integer, float and boolean literals, so just use the rest
                // as-is.
                if let Some((c, span)) = unquote_char(it) {
                    text.push(c);
                    record_span(span);
                } else {
                    let (component, span) =
                        unquote_str(it).unwrap_or_else(|| (it.text.to_string(), it.span));
                    text.push_str(&component);
                    record_span(span);
                }
            }
            // handle boolean literals
            tt::TokenTree::Leaf(tt::Leaf::Ident(id))
                if i % 2 == 0 && (id.text == "true" || id.text == "false") =>
            {
                text.push_str(id.text.as_str());
                record_span(id.span);
            }
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if i % 2 == 1 && punct.char == ',' => (),
            _ => {
                err.get_or_insert(mbe::ExpandError::UnexpectedToken.into());
            }
        }
    }
    let span = span.unwrap_or(tt.delimiter.open);
    ExpandResult { value: quote!(span =>#text), err }
}

fn concat_bytes_expand(
    _db: &dyn ExpandDatabase,
    _arg_id: MacroCallId,
    tt: &tt::Subtree,
    call_site: Span,
) -> ExpandResult<tt::Subtree> {
    let mut bytes = Vec::new();
    let mut err = None;
    let mut span: Option<Span> = None;
    let mut record_span = |s: Span| match &mut span {
        Some(span) if span.anchor == s.anchor => span.range = span.range.cover(s.range),
        Some(_) => (),
        None => span = Some(s),
    };
    for (i, t) in tt.token_trees.iter().enumerate() {
        match t {
            tt::TokenTree::Leaf(tt::Leaf::Literal(lit)) => {
                let token = ast::make::tokens::literal(&lit.to_string());
                record_span(lit.span);
                match token.kind() {
                    syntax::SyntaxKind::BYTE => bytes.push(token.text().to_owned()),
                    syntax::SyntaxKind::BYTE_STRING => {
                        let components = unquote_byte_string(lit).map_or(vec![], |(it, _)| it);
                        components.into_iter().for_each(|it| bytes.push(it.to_string()));
                    }
                    _ => {
                        err.get_or_insert(mbe::ExpandError::UnexpectedToken.into());
                        break;
                    }
                }
            }
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if i % 2 == 1 && punct.char == ',' => (),
            tt::TokenTree::Subtree(tree) if tree.delimiter.kind == tt::DelimiterKind::Bracket => {
                if let Err(e) = concat_bytes_expand_subtree(tree, &mut bytes, &mut record_span) {
                    err.get_or_insert(e);
                    break;
                }
            }
            _ => {
                err.get_or_insert(mbe::ExpandError::UnexpectedToken.into());
                break;
            }
        }
    }
    let value = tt::Subtree {
        delimiter: tt::Delimiter {
            open: call_site,
            close: call_site,
            kind: tt::DelimiterKind::Bracket,
        },
        token_trees: {
            Itertools::intersperse_with(
                bytes.into_iter().map(|it| {
                    tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
                        text: it.into(),
                        span: span.unwrap_or(call_site),
                    }))
                }),
                || {
                    tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct {
                        char: ',',
                        spacing: tt::Spacing::Alone,
                        span: call_site,
                    }))
                },
            )
            .collect()
        },
    };
    ExpandResult { value, err }
}

fn concat_bytes_expand_subtree(
    tree: &tt::Subtree,
    bytes: &mut Vec<String>,
    mut record_span: impl FnMut(Span),
) -> Result<(), ExpandError> {
    for (ti, tt) in tree.token_trees.iter().enumerate() {
        match tt {
            tt::TokenTree::Leaf(tt::Leaf::Literal(it)) => {
                let lit = ast::make::tokens::literal(&it.to_string());
                match lit.kind() {
                    syntax::SyntaxKind::BYTE | syntax::SyntaxKind::INT_NUMBER => {
                        record_span(it.span);
                        bytes.push(lit.text().to_owned())
                    }
                    _ => {
                        return Err(mbe::ExpandError::UnexpectedToken.into());
                    }
                }
            }
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if ti % 2 == 1 && punct.char == ',' => (),
            _ => {
                return Err(mbe::ExpandError::UnexpectedToken.into());
            }
        }
    }
    Ok(())
}

fn concat_idents_expand(
    _db: &dyn ExpandDatabase,
    _arg_id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let mut err = None;
    let mut ident = String::new();
    for (i, t) in tt.token_trees.iter().enumerate() {
        match t {
            tt::TokenTree::Leaf(tt::Leaf::Ident(id)) => {
                ident.push_str(id.text.as_str());
            }
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if i % 2 == 1 && punct.char == ',' => (),
            _ => {
                err.get_or_insert(mbe::ExpandError::UnexpectedToken.into());
            }
        }
    }
    // FIXME merge spans
    let ident = tt::Ident { text: ident.into(), span };
    ExpandResult { value: quote!(span =>#ident), err }
}

fn relative_file(
    db: &dyn ExpandDatabase,
    call_id: MacroCallId,
    path_str: &str,
    allow_recursion: bool,
) -> Result<FileId, ExpandError> {
    let call_site = call_id.as_macro_file().parent(db).original_file_respecting_includes(db);
    let path = AnchoredPath { anchor: call_site, path: path_str };
    let res = db
        .resolve_path(path)
        .ok_or_else(|| ExpandError::other(format!("failed to load file `{path_str}`")))?;
    // Prevent include itself
    if res == call_site && !allow_recursion {
        Err(ExpandError::other(format!("recursive inclusion of `{path_str}`")))
    } else {
        Ok(res)
    }
}

fn parse_string(tt: &tt::Subtree) -> Result<(String, Span), ExpandError> {
    tt.token_trees
        .first()
        .and_then(|tt| match tt {
            tt::TokenTree::Leaf(tt::Leaf::Literal(it)) => unquote_str(it),
            _ => None,
        })
        .ok_or(mbe::ExpandError::ConversionError.into())
}

fn include_expand(
    db: &dyn ExpandDatabase,
    arg_id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let file_id = match include_input_to_file_id(db, arg_id, tt) {
        Ok(it) => it,
        Err(e) => {
            return ExpandResult::new(tt::Subtree::empty(DelimSpan { open: span, close: span }), e)
        }
    };
    match parse_to_token_tree(
        SpanAnchor { file_id, ast_id: ROOT_ERASED_FILE_AST_ID },
        SyntaxContextId::ROOT,
        &db.file_text(file_id),
    ) {
        Some(it) => ExpandResult::ok(it),
        None => ExpandResult::new(
            tt::Subtree::empty(DelimSpan { open: span, close: span }),
            ExpandError::other("failed to parse included file"),
        ),
    }
}

pub fn include_input_to_file_id(
    db: &dyn ExpandDatabase,
    arg_id: MacroCallId,
    arg: &tt::Subtree,
) -> Result<FileId, ExpandError> {
    relative_file(db, arg_id, &parse_string(arg)?.0, false)
}

fn include_bytes_expand(
    _db: &dyn ExpandDatabase,
    _arg_id: MacroCallId,
    _tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    // FIXME: actually read the file here if the user asked for macro expansion
    let res = tt::Subtree {
        delimiter: tt::Delimiter::invisible_spanned(span),
        token_trees: Box::new([tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
            text: r#"b"""#.into(),
            span,
        }))]),
    };
    ExpandResult::ok(res)
}

fn include_str_expand(
    db: &dyn ExpandDatabase,
    arg_id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let (path, span) = match parse_string(tt) {
        Ok(it) => it,
        Err(e) => {
            return ExpandResult::new(tt::Subtree::empty(DelimSpan { open: span, close: span }), e)
        }
    };

    // FIXME: we're not able to read excluded files (which is most of them because
    // it's unusual to `include_str!` a Rust file), but we can return an empty string.
    // Ideally, we'd be able to offer a precise expansion if the user asks for macro
    // expansion.
    let file_id = match relative_file(db, arg_id, &path, true) {
        Ok(file_id) => file_id,
        Err(_) => {
            return ExpandResult::ok(quote!(span =>""));
        }
    };

    let text = db.file_text(file_id);
    let text = &*text;

    ExpandResult::ok(quote!(span =>#text))
}

fn get_env_inner(db: &dyn ExpandDatabase, arg_id: MacroCallId, key: &str) -> Option<String> {
    let krate = db.lookup_intern_macro_call(arg_id).krate;
    db.crate_graph()[krate].env.get(key)
}

fn env_expand(
    db: &dyn ExpandDatabase,
    arg_id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let (key, span) = match parse_string(tt) {
        Ok(it) => it,
        Err(e) => {
            return ExpandResult::new(tt::Subtree::empty(DelimSpan { open: span, close: span }), e)
        }
    };

    let mut err = None;
    let s = get_env_inner(db, arg_id, &key).unwrap_or_else(|| {
        // The only variable rust-analyzer ever sets is `OUT_DIR`, so only diagnose that to avoid
        // unnecessary diagnostics for eg. `CARGO_PKG_NAME`.
        if key == "OUT_DIR" {
            err = Some(ExpandError::other(r#"`OUT_DIR` not set, enable "build scripts" to fix"#));
        }

        // If the variable is unset, still return a dummy string to help type inference along.
        // We cannot use an empty string here, because for
        // `include!(concat!(env!("OUT_DIR"), "/foo.rs"))` will become
        // `include!("foo.rs"), which might go to infinite loop
        "UNRESOLVED_ENV_VAR".to_owned()
    });
    let expanded = quote! {span => #s };

    ExpandResult { value: expanded, err }
}

fn option_env_expand(
    db: &dyn ExpandDatabase,
    arg_id: MacroCallId,
    tt: &tt::Subtree,
    call_site: Span,
) -> ExpandResult<tt::Subtree> {
    let (key, span) = match parse_string(tt) {
        Ok(it) => it,
        Err(e) => {
            return ExpandResult::new(
                tt::Subtree::empty(DelimSpan { open: call_site, close: call_site }),
                e,
            )
        }
    };
    let dollar_crate = dollar_crate(call_site);
    let expanded = match get_env_inner(db, arg_id, &key) {
        None => quote! {call_site => #dollar_crate::option::Option::None::<&str> },
        Some(s) => {
            let s = quote! (span => #s);
            quote! {call_site => #dollar_crate::option::Option::Some(#s) }
        }
    };

    ExpandResult::ok(expanded)
}

fn quote_expand(
    _db: &dyn ExpandDatabase,
    _arg_id: MacroCallId,
    _tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    ExpandResult::new(
        tt::Subtree::empty(tt::DelimSpan { open: span, close: span }),
        ExpandError::other("quote! is not implemented"),
    )
}
