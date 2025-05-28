//! Builtin macro

use base_db::AnchoredPath;
use cfg::CfgExpr;
use either::Either;
use intern::{
    Symbol,
    sym::{self},
};
use mbe::{DelimiterKind, expect_fragment};
use span::{Edition, FileId, Span};
use stdx::format_to;
use syntax::{
    format_smolstr,
    unescape::{Mode, unescape_byte, unescape_char, unescape_unicode},
};
use syntax_bridge::syntax_node_to_token_tree;

use crate::{
    EditionedFileId, ExpandError, ExpandResult, Lookup as _, MacroCallId,
    builtin::quote::{WithDelimiter, dollar_crate, quote},
    db::ExpandDatabase,
    hygiene::{span_with_call_site_ctxt, span_with_def_site_ctxt},
    name,
    span_map::SpanMap,
    tt::{self, DelimSpan, TtElement, TtIter},
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
            fn expander(&self) -> fn (&dyn ExpandDatabase, MacroCallId, &tt::TopSubtree, Span) -> ExpandResult<tt::TopSubtree>  {
                match *self {
                    $( BuiltinFnLikeExpander::$kind => $expand, )*
                }
            }
        }

        impl EagerExpander {
            fn expander(&self) -> fn (&dyn ExpandDatabase, MacroCallId, &tt::TopSubtree, Span) -> ExpandResult<tt::TopSubtree>  {
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
        tt: &tt::TopSubtree,
        span: Span,
    ) -> ExpandResult<tt::TopSubtree> {
        let span = span_with_def_site_ctxt(db, span, id.into(), Edition::CURRENT);
        self.expander()(db, id, tt, span)
    }

    pub fn is_asm(&self) -> bool {
        matches!(self, Self::Asm | Self::GlobalAsm | Self::NakedAsm)
    }
}

impl EagerExpander {
    pub fn expand(
        &self,
        db: &dyn ExpandDatabase,
        id: MacroCallId,
        tt: &tt::TopSubtree,
        span: Span,
    ) -> ExpandResult<tt::TopSubtree> {
        let span = span_with_def_site_ctxt(db, span, id.into(), Edition::CURRENT);
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
    (naked_asm, NakedAsm) => asm_expand,
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
    (concat_bytes, ConcatBytes) => concat_bytes_expand,
    (include, Include) => include_expand,
    (include_bytes, IncludeBytes) => include_bytes_expand,
    (include_str, IncludeStr) => include_str_expand,
    (env, Env) => env_expand,
    (option_env, OptionEnv) => option_env_expand
}

fn mk_pound(span: Span) -> tt::Leaf {
    crate::tt::Leaf::Punct(crate::tt::Punct { char: '#', spacing: crate::tt::Spacing::Alone, span })
}

fn module_path_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    _tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    // Just return a dummy result.
    ExpandResult::ok(quote! {span =>
         "module::path"
    })
}

fn line_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    _tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    // dummy implementation for type-checking purposes
    // Note that `line!` and `column!` will never be implemented properly, as they are by definition
    // not incremental
    ExpandResult::ok(tt::TopSubtree::invisible_from_leaves(
        span,
        [tt::Leaf::Literal(tt::Literal {
            symbol: sym::INTEGER_0,
            span,
            kind: tt::LitKind::Integer,
            suffix: Some(sym::u32),
        })],
    ))
}

fn log_syntax_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    _tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    ExpandResult::ok(quote! {span =>})
}

fn trace_macros_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    _tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    ExpandResult::ok(quote! {span =>})
}

fn stringify_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    let pretty = ::tt::pretty(tt.token_trees().flat_tokens());

    let expanded = quote! {span =>
        #pretty
    };

    ExpandResult::ok(expanded)
}

fn assert_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    let call_site_span = span_with_call_site_ctxt(db, span, id.into(), Edition::CURRENT);

    let mut iter = tt.iter();

    let cond = expect_fragment(
        &mut iter,
        parser::PrefixEntryPoint::Expr,
        id.lookup(db).krate.data(db).edition,
        tt.top_subtree().delimiter.delim_span(),
    );
    _ = iter.expect_char(',');
    let rest = iter.remaining();

    let dollar_crate = dollar_crate(span);
    let panic_args = rest.iter();
    let mac = if use_panic_2021(db, span) {
        quote! {call_site_span => #dollar_crate::panic::panic_2021!(# #panic_args) }
    } else {
        quote! {call_site_span => #dollar_crate::panic!(# #panic_args) }
    };
    let value = cond.value;
    let expanded = quote! {call_site_span =>{
        if !(#value) {
            #mac;
        }
    }};

    match cond.err {
        Some(err) => ExpandResult::new(expanded, err.into()),
        None => ExpandResult::ok(expanded),
    }
}

fn file_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    _tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
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
    tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    let pound = mk_pound(span);
    let mut tt = tt.clone();
    tt.top_subtree_delimiter_mut().kind = tt::DelimiterKind::Parenthesis;
    ExpandResult::ok(quote! {span =>
        builtin #pound format_args #tt
    })
}

fn format_args_nl_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    let pound = mk_pound(span);
    let mut tt = tt.clone();
    tt.top_subtree_delimiter_mut().kind = tt::DelimiterKind::Parenthesis;
    if let Some(tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
        symbol: text,
        kind: tt::LitKind::Str,
        ..
    }))) = tt.0.get_mut(1)
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
    tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    let mut tt = tt.clone();
    tt.top_subtree_delimiter_mut().kind = tt::DelimiterKind::Parenthesis;
    let pound = mk_pound(span);
    let expanded = quote! {span =>
        builtin #pound asm #tt
    };
    ExpandResult::ok(expanded)
}

fn cfg_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    let loc = db.lookup_intern_macro_call(id);
    let expr = CfgExpr::parse(tt);
    let enabled = loc.krate.cfg_options(db).check(&expr) != Some(false);
    let expanded = if enabled { quote!(span=>true) } else { quote!(span=>false) };
    ExpandResult::ok(expanded)
}

fn panic_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    let dollar_crate = dollar_crate(span);
    let call_site_span = span_with_call_site_ctxt(db, span, id.into(), Edition::CURRENT);

    let mac = if use_panic_2021(db, call_site_span) { sym::panic_2021 } else { sym::panic_2015 };

    // Pass the original arguments
    let subtree = WithDelimiter {
        delimiter: tt::Delimiter {
            open: call_site_span,
            close: call_site_span,
            kind: tt::DelimiterKind::Parenthesis,
        },
        token_trees: tt.token_trees(),
    };

    // Expand to a macro call `$crate::panic::panic_{edition}`
    let call = quote!(call_site_span =>#dollar_crate::panic::#mac! #subtree);

    ExpandResult::ok(call)
}

fn unreachable_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    let dollar_crate = dollar_crate(span);
    let call_site_span = span_with_call_site_ctxt(db, span, id.into(), Edition::CURRENT);

    let mac = if use_panic_2021(db, call_site_span) {
        sym::unreachable_2021
    } else {
        sym::unreachable_2015
    };

    // Pass the original arguments
    let mut subtree = tt.clone();
    *subtree.top_subtree_delimiter_mut() = tt::Delimiter {
        open: call_site_span,
        close: call_site_span,
        kind: tt::DelimiterKind::Parenthesis,
    };

    // Expand to a macro call `$crate::panic::panic_{edition}`
    let call = quote!(call_site_span =>#dollar_crate::panic::#mac! #subtree);

    ExpandResult::ok(call)
}

#[allow(clippy::never_loop)]
fn use_panic_2021(db: &dyn ExpandDatabase, span: Span) -> bool {
    // To determine the edition, we check the first span up the expansion
    // stack that does not have #[allow_internal_unstable(edition_panic)].
    // (To avoid using the edition of e.g. the assert!() or debug_assert!() definition.)
    loop {
        let Some(expn) = span.ctx.outer_expn(db) else {
            break false;
        };
        let expn = db.lookup_intern_macro_call(expn.into());
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

fn compile_error_expand(
    _db: &dyn ExpandDatabase,
    _id: MacroCallId,
    tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    let err = match &*tt.0 {
        [
            _,
            tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
                symbol: text,
                span: _,
                kind: tt::LitKind::Str | tt::LitKind::StrRaw(_),
                suffix: _,
            })),
        ] => ExpandError::other(span, Box::from(unescape_str(text).as_str())),
        _ => ExpandError::other(span, "`compile_error!` argument must be a string"),
    };

    ExpandResult { value: quote! {span =>}, err: Some(err) }
}

fn concat_expand(
    _db: &dyn ExpandDatabase,
    _arg_id: MacroCallId,
    tt: &tt::TopSubtree,
    call_site: Span,
) -> ExpandResult<tt::TopSubtree> {
    let mut err = None;
    let mut text = String::new();
    let mut span: Option<Span> = None;
    let mut record_span = |s: Span| match &mut span {
        Some(span) if span.anchor == s.anchor => span.range = span.range.cover(s.range),
        Some(_) => (),
        None => span = Some(s),
    };

    let mut i = 0;
    let mut iter = tt.iter();
    while let Some(mut t) = iter.next() {
        // FIXME: hack on top of a hack: `$e:expr` captures get surrounded in parentheses
        // to ensure the right parsing order, so skip the parentheses here. Ideally we'd
        // implement rustc's model. cc https://github.com/rust-lang/rust-analyzer/pull/10623
        if let TtElement::Subtree(subtree, subtree_iter) = &t {
            if let [tt::TokenTree::Leaf(tt)] = subtree_iter.remaining().flat_tokens() {
                if subtree.delimiter.kind == tt::DelimiterKind::Parenthesis {
                    t = TtElement::Leaf(tt);
                }
            }
        }
        match t {
            TtElement::Leaf(tt::Leaf::Literal(it)) if i % 2 == 0 => {
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
            TtElement::Leaf(tt::Leaf::Ident(id))
                if i % 2 == 0 && (id.sym == sym::true_ || id.sym == sym::false_) =>
            {
                text.push_str(id.sym.as_str());
                record_span(id.span);
            }
            TtElement::Leaf(tt::Leaf::Punct(punct)) if i % 2 == 1 && punct.char == ',' => (),
            // handle negative numbers
            TtElement::Leaf(tt::Leaf::Punct(punct)) if i % 2 == 0 && punct.char == '-' => {
                let t = match iter.next() {
                    Some(t) => t,
                    None => {
                        err.get_or_insert(ExpandError::other(
                            call_site,
                            "unexpected end of input after '-'",
                        ));
                        break;
                    }
                };

                match t {
                    TtElement::Leaf(tt::Leaf::Literal(it))
                        if matches!(it.kind, tt::LitKind::Integer | tt::LitKind::Float) =>
                    {
                        format_to!(text, "-{}", it.symbol.as_str());
                        record_span(punct.span.cover(it.span));
                    }
                    _ => {
                        err.get_or_insert(ExpandError::other(
                            call_site,
                            "expected integer or floating pointer number after '-'",
                        ));
                        break;
                    }
                }
            }
            _ => {
                err.get_or_insert(ExpandError::other(call_site, "unexpected token"));
            }
        }
        i += 1;
    }
    let span = span.unwrap_or_else(|| tt.top_subtree().delimiter.open);
    ExpandResult { value: quote!(span =>#text), err }
}

fn concat_bytes_expand(
    _db: &dyn ExpandDatabase,
    _arg_id: MacroCallId,
    tt: &tt::TopSubtree,
    call_site: Span,
) -> ExpandResult<tt::TopSubtree> {
    let mut bytes = String::new();
    let mut err = None;
    let mut span: Option<Span> = None;
    let mut record_span = |s: Span| match &mut span {
        Some(span) if span.anchor == s.anchor => span.range = span.range.cover(s.range),
        Some(_) => (),
        None => span = Some(s),
    };
    for (i, t) in tt.iter().enumerate() {
        match t {
            TtElement::Leaf(tt::Leaf::Literal(tt::Literal {
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
            TtElement::Leaf(tt::Leaf::Punct(punct)) if i % 2 == 1 && punct.char == ',' => (),
            TtElement::Subtree(tree, tree_iter)
                if tree.delimiter.kind == tt::DelimiterKind::Bracket =>
            {
                if let Err(e) =
                    concat_bytes_expand_subtree(tree_iter, &mut bytes, &mut record_span, call_site)
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
    let span = span.unwrap_or(tt.top_subtree().delimiter.open);
    ExpandResult {
        value: tt::TopSubtree::invisible_from_leaves(
            span,
            [tt::Leaf::Literal(tt::Literal {
                symbol: Symbol::intern(&bytes),
                span,
                kind: tt::LitKind::ByteStr,
                suffix: None,
            })],
        ),
        err,
    }
}

fn concat_bytes_expand_subtree(
    tree_iter: TtIter<'_>,
    bytes: &mut String,
    mut record_span: impl FnMut(Span),
    err_span: Span,
) -> Result<(), ExpandError> {
    for (ti, tt) in tree_iter.enumerate() {
        match tt {
            TtElement::Leaf(tt::Leaf::Literal(tt::Literal {
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
            TtElement::Leaf(tt::Leaf::Literal(tt::Literal {
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
            TtElement::Leaf(tt::Leaf::Punct(punct)) if ti % 2 == 1 && punct.char == ',' => (),
            _ => {
                return Err(ExpandError::other(err_span, "unexpected token"));
            }
        }
    }
    Ok(())
}

fn relative_file(
    db: &dyn ExpandDatabase,
    call_id: MacroCallId,
    path_str: &str,
    allow_recursion: bool,
    err_span: Span,
) -> Result<EditionedFileId, ExpandError> {
    let lookup = db.lookup_intern_macro_call(call_id);
    let call_site = lookup.kind.file_id().original_file_respecting_includes(db).file_id(db);
    let path = AnchoredPath { anchor: call_site, path: path_str };
    let res: FileId = db
        .resolve_path(path)
        .ok_or_else(|| ExpandError::other(err_span, format!("failed to load file `{path_str}`")))?;
    // Prevent include itself
    if res == call_site && !allow_recursion {
        Err(ExpandError::other(err_span, format!("recursive inclusion of `{path_str}`")))
    } else {
        Ok(EditionedFileId::new(db, res, lookup.krate.data(db).edition))
    }
}

fn parse_string(tt: &tt::TopSubtree) -> Result<(Symbol, Span), ExpandError> {
    let delimiter = tt.top_subtree().delimiter;
    tt.iter()
        .next()
        .ok_or(delimiter.open.cover(delimiter.close))
        .and_then(|tt| match tt {
            TtElement::Leaf(tt::Leaf::Literal(tt::Literal {
                symbol: text,
                span,
                kind: tt::LitKind::Str,
                suffix: _,
            })) => Ok((unescape_str(text), *span)),
            TtElement::Leaf(tt::Leaf::Literal(tt::Literal {
                symbol: text,
                span,
                kind: tt::LitKind::StrRaw(_),
                suffix: _,
            })) => Ok((text.clone(), *span)),
            // FIXME: We wrap expression fragments in parentheses which can break this expectation
            // here
            // Remove this once we handle none delims correctly
            TtElement::Subtree(tt, mut tt_iter)
                if tt.delimiter.kind == DelimiterKind::Parenthesis =>
            {
                tt_iter
                    .next()
                    .and_then(|tt| match tt {
                        TtElement::Leaf(tt::Leaf::Literal(tt::Literal {
                            symbol: text,
                            span,
                            kind: tt::LitKind::Str,
                            suffix: _,
                        })) => Some((unescape_str(text), *span)),
                        TtElement::Leaf(tt::Leaf::Literal(tt::Literal {
                            symbol: text,
                            span,
                            kind: tt::LitKind::StrRaw(_),
                            suffix: _,
                        })) => Some((text.clone(), *span)),
                        _ => None,
                    })
                    .ok_or(delimiter.open.cover(delimiter.close))
            }
            TtElement::Leaf(l) => Err(*l.span()),
            TtElement::Subtree(tt, _) => Err(tt.delimiter.open.cover(tt.delimiter.close)),
        })
        .map_err(|span| ExpandError::other(span, "expected string literal"))
}

fn include_expand(
    db: &dyn ExpandDatabase,
    arg_id: MacroCallId,
    tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    let editioned_file_id = match include_input_to_file_id(db, arg_id, tt) {
        Ok(editioned_file_id) => editioned_file_id,
        Err(e) => {
            return ExpandResult::new(
                tt::TopSubtree::empty(DelimSpan { open: span, close: span }),
                e,
            );
        }
    };
    let span_map = db.real_span_map(editioned_file_id);
    // FIXME: Parse errors
    ExpandResult::ok(syntax_node_to_token_tree(
        &db.parse(editioned_file_id).syntax_node(),
        SpanMap::RealSpanMap(span_map),
        span,
        syntax_bridge::DocCommentDesugarMode::ProcMacro,
    ))
}

pub fn include_input_to_file_id(
    db: &dyn ExpandDatabase,
    arg_id: MacroCallId,
    arg: &tt::TopSubtree,
) -> Result<EditionedFileId, ExpandError> {
    let (s, span) = parse_string(arg)?;
    relative_file(db, arg_id, s.as_str(), false, span)
}

fn include_bytes_expand(
    _db: &dyn ExpandDatabase,
    _arg_id: MacroCallId,
    _tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    // FIXME: actually read the file here if the user asked for macro expansion
    let res = tt::TopSubtree::invisible_from_leaves(
        span,
        [tt::Leaf::Literal(tt::Literal {
            symbol: Symbol::empty(),
            span,
            kind: tt::LitKind::ByteStrRaw(1),
            suffix: None,
        })],
    );
    ExpandResult::ok(res)
}

fn include_str_expand(
    db: &dyn ExpandDatabase,
    arg_id: MacroCallId,
    tt: &tt::TopSubtree,
    call_site: Span,
) -> ExpandResult<tt::TopSubtree> {
    let (path, input_span) = match parse_string(tt) {
        Ok(it) => it,
        Err(e) => {
            return ExpandResult::new(
                tt::TopSubtree::empty(DelimSpan { open: call_site, close: call_site }),
                e,
            );
        }
    };

    // FIXME: we're not able to read excluded files (which is most of them because
    // it's unusual to `include_str!` a Rust file), but we can return an empty string.
    // Ideally, we'd be able to offer a precise expansion if the user asks for macro
    // expansion.
    let file_id = match relative_file(db, arg_id, path.as_str(), true, input_span) {
        Ok(file_id) => file_id,
        Err(_) => {
            return ExpandResult::ok(quote!(call_site =>""));
        }
    };

    let text = db.file_text(file_id.file_id(db));
    let text = &*text.text(db);

    ExpandResult::ok(quote!(call_site =>#text))
}

fn get_env_inner(db: &dyn ExpandDatabase, arg_id: MacroCallId, key: &Symbol) -> Option<String> {
    let krate = db.lookup_intern_macro_call(arg_id).krate;
    krate.env(db).get(key.as_str())
}

fn env_expand(
    db: &dyn ExpandDatabase,
    arg_id: MacroCallId,
    tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    let (key, span) = match parse_string(tt) {
        Ok(it) => it,
        Err(e) => {
            return ExpandResult::new(
                tt::TopSubtree::empty(DelimSpan { open: span, close: span }),
                e,
            );
        }
    };

    let mut err = None;
    let s = get_env_inner(db, arg_id, &key).unwrap_or_else(|| {
        // The only variable rust-analyzer ever sets is `OUT_DIR`, so only diagnose that to avoid
        // unnecessary diagnostics for eg. `CARGO_PKG_NAME`.
        if key.as_str() == "OUT_DIR" {
            err = Some(ExpandError::other(
                span,
                r#"`OUT_DIR` not set, build scripts may have failed to run"#,
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
    tt: &tt::TopSubtree,
    call_site: Span,
) -> ExpandResult<tt::TopSubtree> {
    let (key, span) = match parse_string(tt) {
        Ok(it) => it,
        Err(e) => {
            return ExpandResult::new(
                tt::TopSubtree::empty(DelimSpan { open: call_site, close: call_site }),
                e,
            );
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
    _tt: &tt::TopSubtree,
    span: Span,
) -> ExpandResult<tt::TopSubtree> {
    ExpandResult::new(
        tt::TopSubtree::empty(tt::DelimSpan { open: span, close: span }),
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
