//! Builtin macro
use crate::{
    db::AstDatabase, name, quote, AstId, CrateId, EagerMacroId, LazyMacroId, MacroCallId,
    MacroDefId, MacroDefKind, TextSize,
};

use base_db::{AnchoredPath, FileId};
use cfg::CfgExpr;
use either::Either;
use mbe::{parse_exprs_with_sep, parse_to_token_tree, ExpandResult};
use parser::FragmentKind;
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
                id: LazyMacroId,
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
                arg_id: EagerMacroId,
                tt: &tt::Subtree,
            ) -> ExpandResult<Option<(tt::Subtree, FragmentKind)>> {
                let expander = match *self {
                    $( EagerExpander::$e_kind => $e_expand, )*
                };
                expander(db,arg_id,tt)
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

pub fn find_builtin_macro(
    ident: &name::Name,
    krate: CrateId,
    ast_id: AstId<ast::Macro>,
) -> Option<MacroDefId> {
    let kind = find_by_name(ident)?;

    match kind {
        Either::Left(kind) => Some(MacroDefId {
            krate,
            ast_id: Some(ast_id),
            kind: MacroDefKind::BuiltIn(kind),
            local_inner: false,
        }),
        Either::Right(kind) => Some(MacroDefId {
            krate,
            ast_id: Some(ast_id),
            kind: MacroDefKind::BuiltInEager(kind),
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
    // format_args_nl only differs in that it adds a newline in the end,
    // so we use the same stub expansion for now
    (format_args_nl, FormatArgsNl) => format_args_expand,
    (llvm_asm, LlvmAsm) => asm_expand,
    (asm, Asm) => asm_expand,
    (cfg, Cfg) => cfg_expand,

    EAGER:
    (compile_error, CompileError) => compile_error_expand,
    (concat, Concat) => concat_expand,
    (include, Include) => include_expand,
    (include_bytes, IncludeBytes) => include_bytes_expand,
    (include_str, IncludeStr) => include_str_expand,
    (env, Env) => env_expand,
    (option_env, OptionEnv) => option_env_expand
}

fn module_path_expand(
    _db: &dyn AstDatabase,
    _id: LazyMacroId,
    _tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    // Just return a dummy result.
    ExpandResult::ok(quote! { "module::path" })
}

fn line_expand(
    _db: &dyn AstDatabase,
    _id: LazyMacroId,
    _tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    // dummy implementation for type-checking purposes
    let line_num = 0;
    let expanded = quote! {
        #line_num
    };

    ExpandResult::ok(expanded)
}

fn stringify_expand(
    db: &dyn AstDatabase,
    id: LazyMacroId,
    _tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let loc = db.lookup_intern_macro(id);

    let macro_content = {
        let arg = match loc.kind.arg(db) {
            Some(arg) => arg,
            None => return ExpandResult::only_err(mbe::ExpandError::UnexpectedToken),
        };
        let macro_args = arg;
        let text = macro_args.text();
        let without_parens = TextSize::of('(')..text.len() - TextSize::of(')');
        text.slice(without_parens).to_string()
    };

    let expanded = quote! {
        #macro_content
    };

    ExpandResult::ok(expanded)
}

fn column_expand(
    _db: &dyn AstDatabase,
    _id: LazyMacroId,
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
    _id: LazyMacroId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    // A hacky implementation for goto def and hover
    // We expand `assert!(cond, arg1, arg2)` to
    // ```
    // {(cond, &(arg1), &(arg2));}
    // ```,
    // which is wrong but useful.

    let args = parse_exprs_with_sep(tt, ',');

    let arg_tts = args.into_iter().flat_map(|arg| {
        quote! { &(#arg), }
    }.token_trees).collect::<Vec<_>>();

    let expanded = quote! {
        { { (##arg_tts); } }
    };
    ExpandResult::ok(expanded)
}

fn file_expand(
    _db: &dyn AstDatabase,
    _id: LazyMacroId,
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
    _id: LazyMacroId,
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
    }.token_trees).collect::<Vec<_>>();
    let expanded = quote! {
        std::fmt::Arguments::new_v1(&[], &[##arg_tts])
    };
    ExpandResult::ok(expanded)
}

fn asm_expand(
    _db: &dyn AstDatabase,
    _id: LazyMacroId,
    _tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    // both asm and llvm_asm don't return anything, so we can expand them to nothing,
    // for now
    let expanded = quote! {
        ()
    };
    ExpandResult::ok(expanded)
}

fn cfg_expand(
    db: &dyn AstDatabase,
    id: LazyMacroId,
    tt: &tt::Subtree,
) -> ExpandResult<tt::Subtree> {
    let loc = db.lookup_intern_macro(id);
    let expr = CfgExpr::parse(tt);
    let enabled = db.crate_graph()[loc.krate].cfg_options.check(&expr) != Some(false);
    let expanded = if enabled { quote!(true) } else { quote!(false) };
    ExpandResult::ok(expanded)
}

fn unquote_str(lit: &tt::Literal) -> Option<String> {
    let lit = ast::make::tokens::literal(&lit.to_string());
    let token = ast::String::cast(lit)?;
    token.value().map(|it| it.into_owned())
}

fn compile_error_expand(
    _db: &dyn AstDatabase,
    _id: EagerMacroId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<(tt::Subtree, FragmentKind)>> {
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

    ExpandResult { value: Some((quote! {}, FragmentKind::Items)), err: Some(err) }
}

fn concat_expand(
    _db: &dyn AstDatabase,
    _arg_id: EagerMacroId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<(tt::Subtree, FragmentKind)>> {
    let mut err = None;
    let mut text = String::new();
    for (i, t) in tt.token_trees.iter().enumerate() {
        match t {
            tt::TokenTree::Leaf(tt::Leaf::Literal(it)) if i % 2 == 0 => {
                // concat works with string and char literals, so remove any quotes.
                // It also works with integer, float and boolean literals, so just use the rest
                // as-is.
                let component = unquote_str(&it).unwrap_or_else(|| it.text.to_string());
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
    ExpandResult { value: Some((quote!(#text), FragmentKind::Expr)), err }
}

fn relative_file(
    db: &dyn AstDatabase,
    call_id: MacroCallId,
    path: &str,
    allow_recursion: bool,
) -> Option<FileId> {
    let call_site = call_id.as_file().original_file(db);
    let path = AnchoredPath { anchor: call_site, path };
    let res = db.resolve_path(path)?;
    // Prevent include itself
    if res == call_site && !allow_recursion {
        None
    } else {
        Some(res)
    }
}

fn parse_string(tt: &tt::Subtree) -> Result<String, mbe::ExpandError> {
    tt.token_trees
        .get(0)
        .and_then(|tt| match tt {
            tt::TokenTree::Leaf(tt::Leaf::Literal(it)) => unquote_str(&it),
            _ => None,
        })
        .ok_or_else(|| mbe::ExpandError::ConversionError)
}

fn include_expand(
    db: &dyn AstDatabase,
    arg_id: EagerMacroId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<(tt::Subtree, FragmentKind)>> {
    let res = (|| {
        let path = parse_string(tt)?;
        let file_id = relative_file(db, arg_id.into(), &path, false)
            .ok_or_else(|| mbe::ExpandError::ConversionError)?;

        Ok(parse_to_token_tree(&db.file_text(file_id))
            .ok_or_else(|| mbe::ExpandError::ConversionError)?
            .0)
    })();

    match res {
        Ok(res) => {
            // FIXME:
            // Handle include as expression
            ExpandResult::ok(Some((res, FragmentKind::Items)))
        }
        Err(e) => ExpandResult::only_err(e),
    }
}

fn include_bytes_expand(
    _db: &dyn AstDatabase,
    _arg_id: EagerMacroId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<(tt::Subtree, FragmentKind)>> {
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
    ExpandResult::ok(Some((res, FragmentKind::Expr)))
}

fn include_str_expand(
    db: &dyn AstDatabase,
    arg_id: EagerMacroId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<(tt::Subtree, FragmentKind)>> {
    let path = match parse_string(tt) {
        Ok(it) => it,
        Err(e) => return ExpandResult::only_err(e),
    };

    // FIXME: we're not able to read excluded files (which is most of them because
    // it's unusual to `include_str!` a Rust file), but we can return an empty string.
    // Ideally, we'd be able to offer a precise expansion if the user asks for macro
    // expansion.
    let file_id = match relative_file(db, arg_id.into(), &path, true) {
        Some(file_id) => file_id,
        None => {
            return ExpandResult::ok(Some((quote!(""), FragmentKind::Expr)));
        }
    };

    let text = db.file_text(file_id);
    let text = &*text;

    ExpandResult::ok(Some((quote!(#text), FragmentKind::Expr)))
}

fn get_env_inner(db: &dyn AstDatabase, arg_id: EagerMacroId, key: &str) -> Option<String> {
    let krate = db.lookup_intern_eager_expansion(arg_id).krate;
    db.crate_graph()[krate].env.get(key)
}

fn env_expand(
    db: &dyn AstDatabase,
    arg_id: EagerMacroId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<(tt::Subtree, FragmentKind)>> {
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
                r#"`OUT_DIR` not set, enable "load out dirs from check" to fix"#.into(),
            ));
        }

        // If the variable is unset, still return a dummy string to help type inference along.
        // We cannot use an empty string here, because for
        // `include!(concat!(env!("OUT_DIR"), "/foo.rs"))` will become
        // `include!("foo.rs"), which might go to infinite loop
        "__RA_UNIMPLEMENTED__".to_string()
    });
    let expanded = quote! { #s };

    ExpandResult { value: Some((expanded, FragmentKind::Expr)), err }
}

fn option_env_expand(
    db: &dyn AstDatabase,
    arg_id: EagerMacroId,
    tt: &tt::Subtree,
) -> ExpandResult<Option<(tt::Subtree, FragmentKind)>> {
    let key = match parse_string(tt) {
        Ok(it) => it,
        Err(e) => return ExpandResult::only_err(e),
    };

    let expanded = match get_env_inner(db, arg_id, &key) {
        None => quote! { std::option::Option::None::<&str> },
        Some(s) => quote! { std::option::Some(#s) },
    };

    ExpandResult::ok(Some((expanded, FragmentKind::Expr)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        name::AsName, test_db::TestDB, AstNode, EagerCallLoc, MacroCallId, MacroCallKind,
        MacroCallLoc,
    };
    use base_db::{fixture::WithFixture, SourceDatabase};
    use expect_test::{expect, Expect};
    use std::sync::Arc;
    use syntax::ast::NameOwner;

    fn expand_builtin_macro(ra_fixture: &str) -> String {
        let (db, file_id) = TestDB::with_single_file(&ra_fixture);
        let parsed = db.parse(file_id);
        let mut macro_rules: Vec<_> =
            parsed.syntax_node().descendants().filter_map(ast::MacroRules::cast).collect();
        let mut macro_calls: Vec<_> =
            parsed.syntax_node().descendants().filter_map(ast::MacroCall::cast).collect();

        let ast_id_map = db.ast_id_map(file_id.into());

        assert_eq!(macro_rules.len(), 1, "test must contain exactly 1 `macro_rules!`");
        assert_eq!(macro_calls.len(), 1, "test must contain exactly 1 macro call");
        let macro_rules = ast::Macro::from(macro_rules.pop().unwrap());
        let macro_call = macro_calls.pop().unwrap();

        let expander = find_by_name(&macro_rules.name().unwrap().as_name()).unwrap();

        let krate = CrateId(0);
        let file_id = match expander {
            Either::Left(expander) => {
                // the first one should be a macro_rules
                let def = MacroDefId {
                    krate: CrateId(0),
                    ast_id: Some(AstId::new(file_id.into(), ast_id_map.ast_id(&macro_rules))),
                    kind: MacroDefKind::BuiltIn(expander),
                    local_inner: false,
                };

                let loc = MacroCallLoc {
                    def,
                    krate,
                    kind: MacroCallKind::FnLike(AstId::new(
                        file_id.into(),
                        ast_id_map.ast_id(&macro_call),
                    )),
                };

                let id: MacroCallId = db.intern_macro(loc).into();
                id.as_file()
            }
            Either::Right(expander) => {
                // the first one should be a macro_rules
                let def = MacroDefId {
                    krate,
                    ast_id: Some(AstId::new(file_id.into(), ast_id_map.ast_id(&macro_rules))),
                    kind: MacroDefKind::BuiltInEager(expander),
                    local_inner: false,
                };

                let args = macro_call.token_tree().unwrap();
                let parsed_args = mbe::ast_to_token_tree(&args).unwrap().0;
                let call_id = AstId::new(file_id.into(), ast_id_map.ast_id(&macro_call));

                let arg_id = db.intern_eager_expansion({
                    EagerCallLoc {
                        def,
                        fragment: FragmentKind::Expr,
                        subtree: Arc::new(parsed_args.clone()),
                        krate,
                        call: call_id,
                    }
                });

                let (subtree, fragment) = expander.expand(&db, arg_id, &parsed_args).value.unwrap();
                let eager = EagerCallLoc {
                    def,
                    fragment,
                    subtree: Arc::new(subtree),
                    krate,
                    call: call_id,
                };

                let id: MacroCallId = db.intern_eager_expansion(eager).into();
                id.as_file()
            }
        };

        db.parse_or_expand(file_id).unwrap().to_string()
    }

    fn check_expansion(ra_fixture: &str, expect: Expect) {
        let expansion = expand_builtin_macro(ra_fixture);
        expect.assert_eq(&expansion);
    }

    #[test]
    fn test_column_expand() {
        check_expansion(
            r#"
            #[rustc_builtin_macro]
            macro_rules! column {() => {}}
            column!()
            "#,
            expect![["0"]],
        );
    }

    #[test]
    fn test_line_expand() {
        check_expansion(
            r#"
            #[rustc_builtin_macro]
            macro_rules! line {() => {}}
            line!()
            "#,
            expect![["0"]],
        );
    }

    #[test]
    fn test_stringify_expand() {
        check_expansion(
            r#"
            #[rustc_builtin_macro]
            macro_rules! stringify {() => {}}
            stringify!(a b c)
            "#,
            expect![["\"a b c\""]],
        );
    }

    #[test]
    fn test_env_expand() {
        check_expansion(
            r#"
            #[rustc_builtin_macro]
            macro_rules! env {() => {}}
            env!("TEST_ENV_VAR")
            "#,
            expect![["\"__RA_UNIMPLEMENTED__\""]],
        );
    }

    #[test]
    fn test_option_env_expand() {
        check_expansion(
            r#"
            #[rustc_builtin_macro]
            macro_rules! option_env {() => {}}
            option_env!("TEST_ENV_VAR")
            "#,
            expect![["std::option::Option::None:: < &str>"]],
        );
    }

    #[test]
    fn test_file_expand() {
        check_expansion(
            r#"
            #[rustc_builtin_macro]
            macro_rules! file {() => {}}
            file!()
            "#,
            expect![[r#""""#]],
        );
    }

    #[test]
    fn test_assert_expand() {
        check_expansion(
            r#"
            #[rustc_builtin_macro]
            macro_rules! assert {
                ($cond:expr) => ({ /* compiler built-in */ });
                ($cond:expr, $($args:tt)*) => ({ /* compiler built-in */ })
            }
            assert!(true, "{} {:?}", arg1(a, b, c), arg2);
            "#,
            expect![["{{(&(true), &(\"{} {:?}\"), &(arg1(a,b,c)), &(arg2),);}}"]],
        );
    }

    #[test]
    fn test_compile_error_expand() {
        check_expansion(
            r#"
            #[rustc_builtin_macro]
            macro_rules! compile_error {
                ($msg:expr) => ({ /* compiler built-in */ });
                ($msg:expr,) => ({ /* compiler built-in */ })
            }
            compile_error!("error!");
            "#,
            // This expands to nothing (since it's in item position), but emits an error.
            expect![[""]],
        );
    }

    #[test]
    fn test_format_args_expand() {
        check_expansion(
            r#"
            #[rustc_builtin_macro]
            macro_rules! format_args {
                ($fmt:expr) => ({ /* compiler built-in */ });
                ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
            }
            format_args!("{} {:?}", arg1(a, b, c), arg2);
            "#,
            expect![[
                r#"std::fmt::Arguments::new_v1(&[], &[std::fmt::ArgumentV1::new(&(arg1(a,b,c)),std::fmt::Display::fmt),std::fmt::ArgumentV1::new(&(arg2),std::fmt::Display::fmt),])"#
            ]],
        );
    }

    #[test]
    fn test_format_args_expand_with_comma_exprs() {
        check_expansion(
            r#"
            #[rustc_builtin_macro]
            macro_rules! format_args {
                ($fmt:expr) => ({ /* compiler built-in */ });
                ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
            }
            format_args!("{} {:?}", a::<A,B>(), b);
            "#,
            expect![[
                r#"std::fmt::Arguments::new_v1(&[], &[std::fmt::ArgumentV1::new(&(a::<A,B>()),std::fmt::Display::fmt),std::fmt::ArgumentV1::new(&(b),std::fmt::Display::fmt),])"#
            ]],
        );
    }

    #[test]
    fn test_include_bytes_expand() {
        check_expansion(
            r#"
            #[rustc_builtin_macro]
            macro_rules! include_bytes {
                ($file:expr) => {{ /* compiler built-in */ }};
                ($file:expr,) => {{ /* compiler built-in */ }};
            }
            include_bytes("foo");
            "#,
            expect![[r#"b"""#]],
        );
    }

    #[test]
    fn test_concat_expand() {
        check_expansion(
            r##"
            #[rustc_builtin_macro]
            macro_rules! concat {}
            concat!("foo", "r", 0, r#"bar"#, false);
            "##,
            expect![[r#""foor0barfalse""#]],
        );
    }
}
