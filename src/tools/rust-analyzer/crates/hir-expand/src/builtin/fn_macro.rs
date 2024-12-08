//! Builtin macro

use base_db::AnchoredPath;
use cfg::CfgExpr;
use either::Either;
use intern::{sym, Symbol};
use mbe::{expect_fragment, DelimiterKind};
use span::{Edition, EditionedFileId, Span};
use stdx::format_to;
use syntax::{
    format_smolstr,
    unescape::{unescape_byte, unescape_char, unescape_unicode, Mode},
};
use syntax_bridge::syntax_node_to_token_tree;

use crate::{
    builtin::quote::{dollar_crate, quote},
    db::ExpandDatabase,
    hygiene::{span_with_call_site_ctxt, span_with_def_site_ctxt},
    name,
    span_map::SpanMap,
    tt::{self, DelimSpan},
    ExpandError, ExpandResult, HirFileIdExt, Lookup as _, MacroCallId,
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
            fn expander(&self) -> fn (&dyn ExpandDatabase, MacroCallId, &tt::Subtree, Span) -> ExpandResult<tt::Subtree>  {
                match *self {
                    $( BuiltinFnLikeExpander::$kind => $expand, )*
                }
            }
        }

        impl EagerExpander {
            fn expander(&self) -> fn (&dyn ExpandDatabase, MacroCallId, &tt::Subtree, Span) -> ExpandResult<tt::Subtree>  {
                match *self {
                    $( EagerExpander::$e_kind => $e_expand, )*
                }
            }
        }

        fn find_by_name(ident: &name::Name) -> Option<Either<BuiltinFnLikeExpander, EagerExpander>> {
            match ident {
                $( id if id == &sym::$name => Some(Either::Left(BuiltinFnLikeExpander::$kind)), )*
                $( id if id == &sym::$e_name => Some(Either::Right(EagerExpander::$e_kind)), )*
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

    pub fn is_asm(&self) -> bool {
        matches!(self, Self::Asm | Self::GlobalAsm)
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
    (asm, Asm) => asm_expand,
    (global_asm, GlobalAsm) => asm_expand,
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
    crate::builtin::quote::IntoTt::to_subtree(
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
            symbol: sym::INTEGER_0.clone(),
            span,
            kind: tt::LitKind::Integer,
            suffix: Some(sym::u32.clone()),
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

    let mut iter = ::tt::iter::TtIter::new(tt);

    let cond = expect_fragment(
        &mut iter,
        parser::PrefixEntryPoint::Expr,
        db.crate_graph()[id.lookup(db).krate].edition,
        tt::DelimSpan { open: tt.delimiter.open, close: tt.delimiter.close },
    );
    _ = iter.expect_char(',');
    let rest = iter.as_slice();

    let dollar_crate = dollar_crate(span);
    let expanded = match cond.value {
        Some(cond) => {
            let panic_args = rest.iter().cloned();
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
        None => quote! {call_site_span =>{}},
    };

    match cond.err {
        Some(err) => ExpandResult::new(expanded, err.into()),
        None => ExpandResult::ok(expanded),
    }
}

fn file_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    // FIXME: RA purposefully lacks knowledge of absolute file names
    // so just return "".
    let file_name = "file";

    let expanded = quote! {span =>
        #file_name
    };

    ExpandResult::ok(expanded)
}

fn format_args_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let pound = mk_pound(span);
    let mut tt = tt.clone();
    tt.delimiter.kind = tt::DelimiterKind::Parenthesis;
    ExpandResult::ok(quote! {span =>
        builtin #pound format_args #tt
    })
}

fn format_args_nl_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let pound = mk_pound(span);
    let mut tt = tt.clone();
    tt.delimiter.kind = tt::DelimiterKind::Parenthesis;
    if let Some(tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
        symbol: text,
        kind: tt::LitKind::Str,
        ..
    }))) = tt.token_trees.first_mut()
    {
        *text = Symbol::intern(&format_smolstr!("{}\\n", text.as_str()));
    }
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
    let mut tt = tt.clone();
    tt.delimiter.kind = tt::DelimiterKind::Parenthesis;
    let pound = mk_pound(span);
    let expanded = quote! {span =>
        builtin #pound asm #tt
    };
    ExpandResult::ok(expanded)
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

    let mac = if use_panic_2021(db, call_site_span) {
        sym::panic_2021.clone()
    } else {
        sym::panic_2015.clone()
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

fn unreachable_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let dollar_crate = dollar_crate(span);
    let call_site_span = span_with_call_site_ctxt(db, span, id);

    let mac = if use_panic_2021(db, call_site_span) {
        sym::unreachable_2021.clone()
    } else {
        sym::unreachable_2015.clone()
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
        //     if features.iter().any(|&f| f == sym::edition_panic.clone()) {
        //         span = expn.call_site;
        //         continue;
        //     }
        // }
        break expn.def.edition >= Edition::Edition2021;
    }
}

fn compile_error_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
    span: Span,
) -> ExpandResult<tt::Subtree> {
    let err = match &*tt.token_trees {
        [tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
            symbol: text,
            span: _,
            kind: tt::LitKind::Str | tt::LitKind::StrRaw(_),
            suffix: _,
        }))] => ExpandError::other(span, Box::from(unescape_str(text).as_str())),
        _ => ExpandError::other(span, "`compile_error!` argument must be a string"),
    };

    ExpandResult { value: quote! {span =>}, err: Some(err) }
}

fn concat_expand(
    _db: &dyn ExpandDatabase,
    _arg_id: MacroCallId,
    tt: &tt::Subtree,
    call_site: Span,
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
                match it.kind {
                    tt::LitKind::Char => {
                        if let Ok(c) = unescape_char(it.symbol.as_str()) {
                            text.push(c);
                        }
                        record_span(it.span);
                    }
                    tt::LitKind::Integer | tt::LitKind::Float => {
                        format_to!(text, "{}", it.symbol.as_str())
                    }
                    tt::LitKind::Str => {
                        text.push_str(unescape_str(&it.symbol).as_str());
                        record_span(it.span);
                    }
                    tt::LitKind::StrRaw(_) => {
                        format_to!(text, "{}", it.symbol.as_str());
                        record_span(it.span);
                    }
                    tt::LitKind::Byte
                    | tt::LitKind::ByteStr
                    | tt::LitKind::ByteStrRaw(_)
                    | tt::LitKind::CStr
                    | tt::LitKind::CStrRaw(_)
                    | tt::LitKind::Err(_) => {
                        err = Some(ExpandError::other(it.span, "unexpected literal"))
                    }
                }
            }
            // handle boolean literals
            tt::TokenTree::Leaf(tt::Leaf::Ident(id))
                if i % 2 == 0 && (id.sym == sym::true_ || id.sym == sym::false_) =>
            {
                text.push_str(id.sym.as_str());
                record_span(id.span);
            }
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if i % 2 == 1 && punct.char == ',' => (),
            _ => {
                err.get_or_insert(ExpandError::other(call_site, "unexpected token"));
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
    let mut bytes = String::new();
    let mut err = None;
    let mut span: Option<Span> = None;
    let mut record_span = |s: Span| match &mut span {
        Some(span) if span.anchor == s.anchor => span.range = span.range.cover(s.range),
        Some(_) => (),
        None => span = Some(s),
    };
    for (i, t) in tt.token_trees.iter().enumerate() {
        match t {
            tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
                symbol: text,
                span,
                kind,
                suffix: _,
            })) => {
                record_span(*span);
                match kind {
                    tt::LitKind::Byte => {
                        if let Ok(b) = unescape_byte(text.as_str()) {
                            bytes.extend(
                                b.escape_ascii().filter_map(|it| char::from_u32(it as u32)),
                            );
                        }
                    }
                    tt::LitKind::ByteStr => {
                        bytes.push_str(text.as_str());
                    }
                    tt::LitKind::ByteStrRaw(_) => {
                        bytes.extend(text.as_str().escape_debug());
                    }
                    _ => {
                        err.get_or_insert(ExpandError::other(*span, "unexpected token"));
                        break;
                    }
                }
            }
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if i % 2 == 1 && punct.char == ',' => (),
            tt::TokenTree::Subtree(tree) if tree.delimiter.kind == tt::DelimiterKind::Bracket => {
                if let Err(e) =
                    concat_bytes_expand_subtree(tree, &mut bytes, &mut record_span, call_site)
                {
                    err.get_or_insert(e);
                    break;
                }
            }
            _ => {
                err.get_or_insert(ExpandError::other(call_site, "unexpected token"));
                break;
            }
        }
    }
    let span = span.unwrap_or(tt.delimiter.open);
    ExpandResult {
        value: tt::Subtree {
            delimiter: tt::Delimiter::invisible_spanned(span),
            token_trees: vec![tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
                symbol: Symbol::intern(&bytes),
                span,
                kind: tt::LitKind::ByteStr,
                suffix: None,
            }))]
            .into(),
        },
        err,
    }
}

fn concat_bytes_expand_subtree(
    tree: &tt::Subtree,
    bytes: &mut String,
    mut record_span: impl FnMut(Span),
    err_span: Span,
) -> Result<(), ExpandError> {
    for (ti, tt) in tree.token_trees.iter().enumerate() {
        match tt {
            tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
                symbol: text,
                span,
                kind: tt::LitKind::Byte,
                suffix: _,
            })) => {
                if let Ok(b) = unescape_byte(text.as_str()) {
                    bytes.extend(b.escape_ascii().filter_map(|it| char::from_u32(it as u32)));
                }
                record_span(*span);
            }
            tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
                symbol: text,
                span,
                kind: tt::LitKind::Integer,
                suffix: _,
            })) => {
                record_span(*span);
                if let Ok(b) = text.as_str().parse::<u8>() {
                    bytes.extend(b.escape_ascii().filter_map(|it| char::from_u32(it as u32)));
                }
            }
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if ti % 2 == 1 && punct.char == ',' => (),
            _ => {
                return Err(ExpandError::other(err_span, "unexpected token"));
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
                ident.push_str(id.sym.as_str());
            }
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if i % 2 == 1 && punct.char == ',' => (),
            _ => {
                err.get_or_insert(ExpandError::other(span, "unexpected token"));
            }
        }
    }
    // FIXME merge spans
    let ident = tt::Ident { sym: Symbol::intern(&ident), span, is_raw: tt::IdentIsRaw::No };
    ExpandResult { value: quote!(span =>#ident), err }
}

fn relative_file(
    db: &dyn ExpandDatabase,
    call_id: MacroCallId,
    path_str: &str,
    allow_recursion: bool,
    err_span: Span,
) -> Result<EditionedFileId, ExpandError> {
    let lookup = call_id.lookup(db);
    let call_site = lookup.kind.file_id().original_file_respecting_includes(db).file_id();
    let path = AnchoredPath { anchor: call_site, path: path_str };
    let res = db
        .resolve_path(path)
        .ok_or_else(|| ExpandError::other(err_span, format!("failed to load file `{path_str}`")))?;
    // Prevent include itself
    if res == call_site && !allow_recursion {
        Err(ExpandError::other(err_span, format!("recursive inclusion of `{path_str}`")))
    } else {
        Ok(EditionedFileId::new(res, db.crate_graph()[lookup.krate].edition))
    }
}

fn parse_string(tt: &tt::Subtree) -> Result<(Symbol, Span), ExpandError> {
    tt.token_trees
        .first()
        .ok_or(tt.delimiter.open.cover(tt.delimiter.close))
        .and_then(|tt| match tt {
            tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
                symbol: text,
                span,
                kind: tt::LitKind::Str,
                suffix: _,
            })) => Ok((unescape_str(text), *span)),
            tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
                symbol: text,
                span,
                kind: tt::LitKind::StrRaw(_),
                suffix: _,
            })) => Ok((text.clone(), *span)),
            // FIXME: We wrap expression fragments in parentheses which can break this expectation
            // here
            // Remove this once we handle none delims correctly
            tt::TokenTree::Subtree(tt) if tt.delimiter.kind == DelimiterKind::Parenthesis => {
                tt.token_trees.first().and_then(|tt| match tt {
                    tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
                        symbol: text,
                        span,
                        kind: tt::LitKind::Str,
                        suffix: _,
                    })) => Some((unescape_str(text), *span)),
                    tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
                        symbol: text,
                        span,
                        kind: tt::LitKind::StrRaw(_),
                        suffix: _,
                    })) => Some((text.clone(), *span)),
                    _ => None,
                })
            }
            .ok_or(tt.delimiter.open.cover(tt.delimiter.close)),
            ::tt::TokenTree::Leaf(l) => Err(*l.span()),
            ::tt::TokenTree::Subtree(tt) => Err(tt.delimiter.open.cover(tt.delimiter.close)),
        })
        .map_err(|span| ExpandError::other(span, "expected string literal"))
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
    let span_map = db.real_span_map(file_id);
    // FIXME: Parse errors
    ExpandResult::ok(syntax_node_to_token_tree(
        &db.parse(file_id).syntax_node(),
        SpanMap::RealSpanMap(span_map),
        span,
        syntax_bridge::DocCommentDesugarMode::ProcMacro,
    ))
}

pub fn include_input_to_file_id(
    db: &dyn ExpandDatabase,
    arg_id: MacroCallId,
    arg: &tt::Subtree,
) -> Result<EditionedFileId, ExpandError> {
    let (s, span) = parse_string(arg)?;
    relative_file(db, arg_id, s.as_str(), false, span)
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
            symbol: Symbol::empty(),
            span,
            kind: tt::LitKind::ByteStrRaw(1),
            suffix: None,
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
    let file_id = match relative_file(db, arg_id, path.as_str(), true, span) {
        Ok(file_id) => file_id,
        Err(_) => {
            return ExpandResult::ok(quote!(span =>""));
        }
    };

    let text = db.file_text(file_id.file_id());
    let text = &*text;

    ExpandResult::ok(quote!(span =>#text))
}

fn get_env_inner(db: &dyn ExpandDatabase, arg_id: MacroCallId, key: &Symbol) -> Option<String> {
    let krate = db.lookup_intern_macro_call(arg_id).krate;
    db.crate_graph()[krate].env.get(key.as_str())
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
        if key.as_str() == "OUT_DIR" {
            err = Some(ExpandError::other(
                span,
                r#"`OUT_DIR` not set, enable "build scripts" to fix"#,
            ));
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
        ExpandError::other(span, "quote! is not implemented"),
    )
}

fn unescape_str(s: &Symbol) -> Symbol {
    if s.as_str().contains('\\') {
        let s = s.as_str();
        let mut buf = String::with_capacity(s.len());
        unescape_unicode(s, Mode::Str, &mut |_, c| {
            if let Ok(c) = c {
                buf.push(c)
            }
        });
        Symbol::intern(&buf)
    } else {
        s.clone()
    }
}
