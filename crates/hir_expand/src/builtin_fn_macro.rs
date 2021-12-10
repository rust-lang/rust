//! Builtin macro
use crate::{
    db::AstDatabase, name, quote, AstId, CrateId, MacroCallId, MacroCallLoc, MacroDefId,
    MacroDefKind,
};

use base_db::{AnchoredPath, Edition, FileId};
use cfg::CfgExpr;
use either::Either;
use mbe::{parse_exprs_with_sep, parse_to_token_tree, ExpandResult};
use syntax::ast::{self, AstToken};

macro_rules! register_builtin {
    ( LAZY: $(($name:ident, $kind: ident) => $expand:ident),* , EAGER: $(($e_name:ident, $e_kind: ident) => $e_expand:ident),*  ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinFnLikeExpander {
            $($kind),*
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum EagerExpander {
            $($e_kind),*
        }

        impl BuiltinFnLikeExpander {
            pub fn expand(
                &self,
                db: &dyn AstDatabase,
                id: MacroCallId,
                tt: &tt::Subtree,
            ) -> ExpandResult<tt::Subtree> {
                let expander = match *self {
                    $( BuiltinFnLikeExpander::$kind => $expand, )*
                };
                expander(db, id, tt)
            }
        }

        impl EagerExpander {
            pub fn expand(
                &self,
                db: &dyn AstDatabase,
                arg_id: MacroCallId,
                tt: &tt::Subtree,
            ) -> ExpandResult<Option<ExpandedEager>> {
                let expander = match *self {
                    $( EagerExpander::$e_kind => $e_expand, )*
                };
                expander(db, arg_id, tt)
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

#[derive(Debug)]
pub struct ExpandedEager {
    pub(crate) subtree: tt::Subtree,
    /// The included file ID of the include macro.
    pub(crate) included_file: Option<FileId>,
}

impl ExpandedEager {
    fn new(subtree: tt::Subtree) -> Self {
        ExpandedEager { subtree, included_file: None }
    }
}

pub fn find_builtin_macro(
    ident: &name::Name,
    krate: CrateId,
    ast_id: AstId<ast::Macro>,
) -> Option<MacroDefId> {
    let kind = find_by_name(ident)?;

    match kind {
        Either::Left(kind) => Some(MacroDefId {
            krate,
            kind: MacroDefKind::BuiltIn(kind, ast_id),
            local_inner: false,
        }),
        Either::Right(kind) => Some(MacroDefId {
            krate,
            kind: MacroDefKind::BuiltInEager(kind, ast_id),
            local_inner: false,
        }),
    }
}

register_builtin! {
    LAZY:
    (column, Column) => column_expand,
    (file, File) => file_expand,
    (line, Line) => line_expand,
    (module_path, ModulePath) => module_path_expand,
    (assert, Assert) => assert_expand,
    (stringify, Stringify) => stringify_expand,
    (format_args, FormatArgs) => format_args_expand,
    (const_format_args, ConstFormatArgs) => format_args_expand,
    // format_args_nl only differs in that it adds a newline in the end,
    // so we use the same stub expansion for now
    (format_args_nl, FormatArgsNl) => format_args_expand,
    (llvm_asm, LlvmAsm) => asm_expand,
    (asm, Asm) => asm_expand,
    (global_asm, GlobalAsm) => global_asm_expand,
    (cfg, Cfg) => cfg_expand,
    (core_panic, CorePanic) => panic_expand,
    (std_panic, StdPanic) => panic_expand,
    (log_syntax, LogSyntax) => log_syntax_expand,
    (trace_macros, TraceMacros) => trace_macros_expand,

    EAGER:
    (compile_error, CompileError) => compile_error_expand,
    (concat, Concat) => concat_expand,
    (concat_idents, ConcatIdents) => concat_idents_expand,
    (include, Include) => include_expand,
    (include_bytes, IncludeBytes) => include_bytes_expand,
    (include_str, IncludeStr) => include_str_expand,
    (env, Env) => env_expand,
    (option_env, OptionEnv) => option_env_expand
}

fn module_path_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    // Just return a dummy result.
    ExpandResult::ok(quote! { "module::path" })
}

fn line_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    // dummy implementation for type-checking purposes
    let line_num = 0;
    let expanded = quote! {
        #line_num
    };

    ExpandResult::ok(expanded)
}

fn log_syntax_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    ExpandResult::ok(quote! {})
}

fn trace_macros_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    ExpandResult::ok(quote! {})
}

fn stringify_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let pretty = tt::pretty(&tt.token_trees);

    let expanded = quote! {
        #pretty
    };

    ExpandResult::ok(expanded)
}

fn column_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    // dummy implementation for type-checking purposes
    let col_num = 0;
    let expanded = quote! {
        #col_num
    };

    ExpandResult::ok(expanded)
}

fn assert_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let krate = tt::Ident { text: "$crate".into(), id: tt::TokenId::unspecified() };
    let args = parse_exprs_with_sep(tt, ',');
    let expanded = match &*args {
        [cond, panic_args @ ..] => {
            let comma = tt::Subtree {
                delimiter: None,
                token_trees: vec![tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct {
                    char: ',',
                    spacing: tt::Spacing::Alone,
                    id: tt::TokenId::unspecified(),
                }))],
            };
            let cond = cond.clone();
            let panic_args = itertools::Itertools::intersperse(panic_args.iter().cloned(), comma);
            quote! {{
                if !#cond {
                    #krate::panic!(##panic_args);
                }
            }}
        }
        [] => quote! {{}},
    };

    ExpandResult::ok(expanded)
}

fn file_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    // FIXME: RA purposefully lacks knowledge of absolute file names
    // so just return "".
    let file_name = "";

    let expanded = quote! {
        #file_name
    };

    ExpandResult::ok(expanded)
}

fn format_args_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    // We expand `format_args!("", a1, a2)` to
    // ```
    // std::fmt::Arguments::new_v1(&[], &[
    //   std::fmt::ArgumentV1::new(&arg1,std::fmt::Display::fmt),
    //   std::fmt::ArgumentV1::new(&arg2,std::fmt::Display::fmt),
    // ])
    // ```,
    // which is still not really correct, but close enough for now
    let mut args = parse_exprs_with_sep(tt, ',');

    if args.is_empty() {
        return ExpandResult::only_err(mbe::ExpandError::NoMatchingRule);
    }
    for arg in &mut args {
        // Remove `key =`.
        if matches!(arg.token_trees.get(1), Some(tt::TokenTree::Leaf(tt::Leaf::Punct(p))) if p.char == '=' && p.spacing != tt::Spacing::Joint)
        {
            arg.token_trees.drain(..2);
        }
    }
    let _format_string = args.remove(0);
    let arg_tts = args.into_iter().flat_map(|arg| {
        quote! { std::fmt::ArgumentV1::new(&(#arg), std::fmt::Display::fmt), }
    }.token_trees);
    let expanded = quote! {
        // It's unsafe since https://github.com/rust-lang/rust/pull/83302
        // Wrap an unsafe block to avoid false-positive `missing-unsafe` lint.
        // FIXME: Currently we don't have `unused_unsafe` lint so an extra unsafe block won't cause issues on early
        // stable rust-src.
        unsafe {
            std::fmt::Arguments::new_v1(&[], &[##arg_tts])
        }
    };
    ExpandResult::ok(expanded)
}

fn asm_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    // We expand all assembly snippets to `format_args!` invocations to get format syntax
    // highlighting for them.

    let krate = tt::Ident { text: "$crate".into(), id: tt::TokenId::unspecified() };

    let mut literals = Vec::new();
    for tt in tt.token_trees.chunks(2) {
        match tt {
            [tt::TokenTree::Leaf(tt::Leaf::Literal(lit))]
            | [tt::TokenTree::Leaf(tt::Leaf::Literal(lit)), tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: ',', id: _, spacing: _ }))] =>
            {
                let krate = krate.clone();
                literals.push(quote!(#krate::format_args!(#lit);));
            }
            _ => break,
        }
    }

    let expanded = quote! {{
        ##literals
        ()
    }};
    ExpandResult::ok(expanded)
}

fn global_asm_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    // Expand to nothing (at item-level)
    ExpandResult::ok(quote! {})
}

fn cfg_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let loc = db.lookup_intern_macro_call(id);
    let expr = CfgExpr::parse(tt);
    let enabled = db.crate_graph()[loc.krate].cfg_options.check(&expr) != Some(false);
    let expanded = if enabled { quote!(true) } else { quote!(false) };
    ExpandResult::ok(expanded)
}

fn panic_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let loc: MacroCallLoc = db.lookup_intern_macro_call(id);
    // Expand to a macro call `$crate::panic::panic_{edition}`
    let krate = tt::Ident { text: "$crate".into(), id: tt::TokenId::unspecified() };
    let mut call = if db.crate_graph()[loc.krate].edition == Edition::Edition2021 {
        quote!(#krate::panic::panic_2021!)
    } else {
        quote!(#krate::panic::panic_2015!)
    };

    // Pass the original arguments
    call.token_trees.push(tt::TokenTree::Subtree(tt.clone()));
    ExpandResult::ok(call)
}

fn unquote_str(lit: &tt::Literal) -> Option<String> {
    let lit = ast::make::tokens::literal(&lit.to_string());
    let token = ast::String::cast(lit)?;
    token.value().map(|it| it.into_owned())
}

fn compile_error_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<ExpandedEager>> {
    let err = match &*tt.token_trees {
        [tt::TokenTree::Leaf(tt::Leaf::Literal(it))] => {
            let text = it.text.as_str();
            if text.starts_with('"') && text.ends_with('"') {
                // FIXME: does not handle raw strings
                mbe::ExpandError::Other(text[1..text.len() - 1].to_string())
            } else {
                mbe::ExpandError::BindingError("`compile_error!` argument must be a string".into())
            }
        }
        _ => mbe::ExpandError::BindingError("`compile_error!` argument must be a string".into()),
    };

    ExpandResult { value: Some(ExpandedEager::new(quote! {})), err: Some(err) }
}

fn concat_expand(
    _db: &dyn AstDatabase,
    _arg_id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<ExpandedEager>> {
    let mut err = None;
    let mut text = String::new();
    for (i, mut t) in tt.token_trees.iter().enumerate() {
        // FIXME: hack on top of a hack: `$e:expr` captures get surrounded in parentheses
        // to ensure the right parsing order, so skip the parentheses here. Ideally we'd
        // implement rustc's model. cc https://github.com/rust-analyzer/rust-analyzer/pull/10623
        if let tt::TokenTree::Subtree(tt::Subtree { delimiter: Some(delim), token_trees }) = t {
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
                let component = unquote_str(it).unwrap_or_else(|| it.text.to_string());
                text.push_str(&component);
            }
            // handle boolean literals
            tt::TokenTree::Leaf(tt::Leaf::Ident(id))
                if i % 2 == 0 && (id.text == "true" || id.text == "false") =>
            {
                text.push_str(id.text.as_str());
            }
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if i % 2 == 1 && punct.char == ',' => (),
            _ => {
                err.get_or_insert(mbe::ExpandError::UnexpectedToken);
            }
        }
    }
    ExpandResult { value: Some(ExpandedEager::new(quote!(#text))), err }
}

fn concat_idents_expand(
    _db: &dyn AstDatabase,
    _arg_id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<ExpandedEager>> {
    let mut err = None;
    let mut ident = String::new();
    for (i, t) in tt.token_trees.iter().enumerate() {
        match t {
            tt::TokenTree::Leaf(tt::Leaf::Ident(id)) => {
                ident.push_str(id.text.as_str());
            }
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if i % 2 == 1 && punct.char == ',' => (),
            _ => {
                err.get_or_insert(mbe::ExpandError::UnexpectedToken);
            }
        }
    }
    let ident = tt::Ident { text: ident.into(), id: tt::TokenId::unspecified() };
    ExpandResult { value: Some(ExpandedEager::new(quote!(#ident))), err }
}

fn relative_file(
    db: &dyn AstDatabase,
    call_id: MacroCallId,
    path_str: &str,
    allow_recursion: bool,
) -> Result<FileId, mbe::ExpandError> {
    let call_site = call_id.as_file().original_file(db);
    let path = AnchoredPath { anchor: call_site, path: path_str };
    let res = db
        .resolve_path(path)
        .ok_or_else(|| mbe::ExpandError::Other(format!("failed to load file `{}`", path_str)))?;
    // Prevent include itself
    if res == call_site && !allow_recursion {
        Err(mbe::ExpandError::Other(format!("recursive inclusion of `{}`", path_str)))
    } else {
        Ok(res)
    }
}

fn parse_string(tt: &tt::Subtree) -> Result<String, mbe::ExpandError> {
    tt.token_trees
        .get(0)
        .and_then(|tt| match tt {
            tt::TokenTree::Leaf(tt::Leaf::Literal(it)) => unquote_str(it),
            _ => None,
        })
        .ok_or(mbe::ExpandError::ConversionError)
}

fn include_expand(
    db: &dyn AstDatabase,
    arg_id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<ExpandedEager>> {
    let res = (|| {
        let path = parse_string(tt)?;
        let file_id = relative_file(db, arg_id, &path, false)?;

        let subtree =
            parse_to_token_tree(&db.file_text(file_id)).ok_or(mbe::ExpandError::ConversionError)?.0;
        Ok((subtree, file_id))
    })();

    match res {
        Ok((subtree, file_id)) => {
            ExpandResult::ok(Some(ExpandedEager { subtree, included_file: Some(file_id) }))
        }
        Err(e) => ExpandResult::only_err(e),
    }
}

fn include_bytes_expand(
    _db: &dyn AstDatabase,
    _arg_id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<ExpandedEager>> {
    if let Err(e) = parse_string(tt) {
        return ExpandResult::only_err(e);
    }

    // FIXME: actually read the file here if the user asked for macro expansion
    let res = tt::Subtree {
        delimiter: None,
        token_trees: vec![tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
            text: r#"b"""#.into(),
            id: tt::TokenId::unspecified(),
        }))],
    };
    ExpandResult::ok(Some(ExpandedEager::new(res)))
}

fn include_str_expand(
    db: &dyn AstDatabase,
    arg_id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<ExpandedEager>> {
    let path = match parse_string(tt) {
        Ok(it) => it,
        Err(e) => return ExpandResult::only_err(e),
    };

    // FIXME: we're not able to read excluded files (which is most of them because
    // it's unusual to `include_str!` a Rust file), but we can return an empty string.
    // Ideally, we'd be able to offer a precise expansion if the user asks for macro
    // expansion.
    let file_id = match relative_file(db, arg_id, &path, true) {
        Ok(file_id) => file_id,
        Err(_) => {
            return ExpandResult::ok(Some(ExpandedEager::new(quote!(""))));
        }
    };

    let text = db.file_text(file_id);
    let text = &*text;

    ExpandResult::ok(Some(ExpandedEager::new(quote!(#text))))
}

fn get_env_inner(db: &dyn AstDatabase, arg_id: MacroCallId, key: &str) -> Option<String> {
    let krate = db.lookup_intern_macro_call(arg_id).krate;
    db.crate_graph()[krate].env.get(key)
}

fn env_expand(
    db: &dyn AstDatabase,
    arg_id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<ExpandedEager>> {
    let key = match parse_string(tt) {
        Ok(it) => it,
        Err(e) => return ExpandResult::only_err(e),
    };

    let mut err = None;
    let s = get_env_inner(db, arg_id, &key).unwrap_or_else(|| {
        // The only variable rust-analyzer ever sets is `OUT_DIR`, so only diagnose that to avoid
        // unnecessary diagnostics for eg. `CARGO_PKG_NAME`.
        if key == "OUT_DIR" {
            err = Some(mbe::ExpandError::Other(
                r#"`OUT_DIR` not set, enable "run build scripts" to fix"#.into(),
            ));
        }

        // If the variable is unset, still return a dummy string to help type inference along.
        // We cannot use an empty string here, because for
        // `include!(concat!(env!("OUT_DIR"), "/foo.rs"))` will become
        // `include!("foo.rs"), which might go to infinite loop
        "__RA_UNIMPLEMENTED__".to_string()
    });
    let expanded = quote! { #s };

    ExpandResult { value: Some(ExpandedEager::new(expanded)), err }
}

fn option_env_expand(
    db: &dyn AstDatabase,
    arg_id: MacroCallId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<ExpandedEager>> {
    let key = match parse_string(tt) {
        Ok(it) => it,
        Err(e) => return ExpandResult::only_err(e),
    };

    let expanded = match get_env_inner(db, arg_id, &key) {
        None => quote! { std::option::Option::None::<&str> },
        Some(s) => quote! { std::option::Some(#s) },
    };

    ExpandResult::ok(Some(ExpandedEager::new(expanded)))
}
