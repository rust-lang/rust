//! Builtin macro
use crate::db::AstDatabase;
use crate::{
    ast::{self, AstToken, HasStringValue},
    name, AstId, CrateId, MacroDefId, MacroDefKind, TextUnit,
};

use crate::{quote, EagerMacroId, LazyMacroId, MacroCallId};
use either::Either;
use ra_db::{FileId, RelativePath};
use ra_parser::FragmentKind;

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
            ) -> Result<tt::Subtree, mbe::ExpandError> {
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
            ) -> Result<(tt::Subtree, FragmentKind), mbe::ExpandError> {
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
    ast_id: AstId<ast::MacroCall>,
) -> Option<MacroDefId> {
    let kind = find_by_name(ident)?;

    match kind {
        Either::Left(kind) => Some(MacroDefId {
            krate: Some(krate),
            ast_id: Some(ast_id),
            kind: MacroDefKind::BuiltIn(kind),
        }),
        Either::Right(kind) => Some(MacroDefId {
            krate: Some(krate),
            ast_id: Some(ast_id),
            kind: MacroDefKind::BuiltInEager(kind),
        }),
    }
}

register_builtin! {
    LAZY:
    (column, Column) => column_expand,
    (compile_error, CompileError) => compile_error_expand,
    (file, File) => file_expand,
    (line, Line) => line_expand,
    (stringify, Stringify) => stringify_expand,
    (format_args, FormatArgs) => format_args_expand,
    (env, Env) => env_expand,
    (option_env, OptionEnv) => option_env_expand,
    // format_args_nl only differs in that it adds a newline in the end,
    // so we use the same stub expansion for now
    (format_args_nl, FormatArgsNl) => format_args_expand,

    EAGER:
    (concat, Concat) => concat_expand,
    (include, Include) => include_expand
}

fn line_expand(
    _db: &dyn AstDatabase,
    _id: LazyMacroId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    // dummy implementation for type-checking purposes
    let line_num = 0;
    let expanded = quote! {
        #line_num
    };

    Ok(expanded)
}

fn stringify_expand(
    db: &dyn AstDatabase,
    id: LazyMacroId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    let loc = db.lookup_intern_macro(id);

    let macro_content = {
        let arg = loc.kind.arg(db).ok_or_else(|| mbe::ExpandError::UnexpectedToken)?;
        let macro_args = arg;
        let text = macro_args.text();
        let without_parens = TextUnit::of_char('(')..text.len() - TextUnit::of_char(')');
        text.slice(without_parens).to_string()
    };

    let expanded = quote! {
        #macro_content
    };

    Ok(expanded)
}

fn env_expand(
    _db: &dyn AstDatabase,
    _id: LazyMacroId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    // dummy implementation for type-checking purposes
    let expanded = quote! { "" };

    Ok(expanded)
}

fn option_env_expand(
    _db: &dyn AstDatabase,
    _id: LazyMacroId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    // dummy implementation for type-checking purposes
    let expanded = quote! { std::option::Option::None::<&str> };

    Ok(expanded)
}

fn column_expand(
    _db: &dyn AstDatabase,
    _id: LazyMacroId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    // dummy implementation for type-checking purposes
    let col_num = 0;
    let expanded = quote! {
        #col_num
    };

    Ok(expanded)
}

fn file_expand(
    _db: &dyn AstDatabase,
    _id: LazyMacroId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    // FIXME: RA purposefully lacks knowledge of absolute file names
    // so just return "".
    let file_name = "";

    let expanded = quote! {
        #file_name
    };

    Ok(expanded)
}

fn compile_error_expand(
    _db: &dyn AstDatabase,
    _id: LazyMacroId,
    tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    if tt.count() == 1 {
        if let tt::TokenTree::Leaf(tt::Leaf::Literal(it)) = &tt.token_trees[0] {
            let s = it.text.as_str();
            if s.contains('"') {
                return Ok(quote! { loop { #it }});
            }
        };
    }

    Err(mbe::ExpandError::BindingError("Must be a string".into()))
}

fn format_args_expand(
    _db: &dyn AstDatabase,
    _id: LazyMacroId,
    tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    // We expand `format_args!("", a1, a2)` to
    // ```
    // std::fmt::Arguments::new_v1(&[], &[
    //   std::fmt::ArgumentV1::new(&arg1,std::fmt::Display::fmt),
    //   std::fmt::ArgumentV1::new(&arg2,std::fmt::Display::fmt),
    // ])
    // ```,
    // which is still not really correct, but close enough for now
    let mut args = Vec::new();
    let mut current = Vec::new();
    for tt in tt.token_trees.iter().cloned() {
        match tt {
            tt::TokenTree::Leaf(tt::Leaf::Punct(p)) if p.char == ',' => {
                args.push(current);
                current = Vec::new();
            }
            _ => {
                current.push(tt);
            }
        }
    }
    if !current.is_empty() {
        args.push(current);
    }
    if args.is_empty() {
        return Err(mbe::ExpandError::NoMatchingRule);
    }
    let _format_string = args.remove(0);
    let arg_tts = args.into_iter().flat_map(|arg| {
        quote! { std::fmt::ArgumentV1::new(&(##arg), std::fmt::Display::fmt), }
    }.token_trees).collect::<Vec<_>>();
    let expanded = quote! {
        std::fmt::Arguments::new_v1(&[], &[##arg_tts])
    };
    Ok(expanded)
}

fn unquote_str(lit: &tt::Literal) -> Option<String> {
    let lit = ast::make::tokens::literal(&lit.to_string());
    let token = ast::String::cast(lit)?;
    token.value()
}

fn concat_expand(
    _db: &dyn AstDatabase,
    _arg_id: EagerMacroId,
    tt: &tt::Subtree,
) -> Result<(tt::Subtree, FragmentKind), mbe::ExpandError> {
    let mut text = String::new();
    for (i, t) in tt.token_trees.iter().enumerate() {
        match t {
            tt::TokenTree::Leaf(tt::Leaf::Literal(it)) if i % 2 == 0 => {
                text += &unquote_str(&it).ok_or_else(|| mbe::ExpandError::ConversionError)?;
            }
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if i % 2 == 1 && punct.char == ',' => (),
            _ => return Err(mbe::ExpandError::UnexpectedToken),
        }
    }

    Ok((quote!(#text), FragmentKind::Expr))
}

fn relative_file(db: &dyn AstDatabase, call_id: MacroCallId, path: &str) -> Option<FileId> {
    let call_site = call_id.as_file().original_file(db);
    let path = RelativePath::new(&path);

    db.resolve_relative_path(call_site, &path)
}

fn include_expand(
    db: &dyn AstDatabase,
    arg_id: EagerMacroId,
    tt: &tt::Subtree,
) -> Result<(tt::Subtree, FragmentKind), mbe::ExpandError> {
    let path = tt
        .token_trees
        .get(0)
        .and_then(|tt| match tt {
            tt::TokenTree::Leaf(tt::Leaf::Literal(it)) => unquote_str(&it),
            _ => None,
        })
        .ok_or_else(|| mbe::ExpandError::ConversionError)?;

    let file_id =
        relative_file(db, arg_id.into(), &path).ok_or_else(|| mbe::ExpandError::ConversionError)?;

    // FIXME:
    // Handle include as expression
    let node =
        db.parse_or_expand(file_id.into()).ok_or_else(|| mbe::ExpandError::ConversionError)?;
    let res =
        mbe::syntax_node_to_token_tree(&node).ok_or_else(|| mbe::ExpandError::ConversionError)?.0;

    Ok((res, FragmentKind::Items))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{name::AsName, test_db::TestDB, AstNode, MacroCallId, MacroCallKind, MacroCallLoc};
    use ra_db::{fixture::WithFixture, SourceDatabase};
    use ra_syntax::ast::NameOwner;

    fn expand_builtin_macro(ra_fixture: &str) -> String {
        let (db, file_id) = TestDB::with_single_file(&ra_fixture);
        let parsed = db.parse(file_id);
        let macro_calls: Vec<_> =
            parsed.syntax_node().descendants().filter_map(ast::MacroCall::cast).collect();

        let ast_id_map = db.ast_id_map(file_id.into());

        let expander = find_by_name(&macro_calls[0].name().unwrap().as_name()).unwrap();
        let expander = expander.left().unwrap();

        // the first one should be a macro_rules
        let def = MacroDefId {
            krate: Some(CrateId(0)),
            ast_id: Some(AstId::new(file_id.into(), ast_id_map.ast_id(&macro_calls[0]))),
            kind: MacroDefKind::BuiltIn(expander),
        };

        let loc = MacroCallLoc {
            def,
            kind: MacroCallKind::FnLike(AstId::new(
                file_id.into(),
                ast_id_map.ast_id(&macro_calls[1]),
            )),
        };

        let id: MacroCallId = db.intern_macro(loc).into();
        let parsed = db.parse_or_expand(id.as_file()).unwrap();

        parsed.text().to_string()
    }

    #[test]
    fn test_column_expand() {
        let expanded = expand_builtin_macro(
            r#"
            #[rustc_builtin_macro]
            macro_rules! column {() => {}}
            column!()
            "#,
        );

        assert_eq!(expanded, "0");
    }

    #[test]
    fn test_line_expand() {
        let expanded = expand_builtin_macro(
            r#"
            #[rustc_builtin_macro]
            macro_rules! line {() => {}}
            line!()
            "#,
        );

        assert_eq!(expanded, "0");
    }

    #[test]
    fn test_stringify_expand() {
        let expanded = expand_builtin_macro(
            r#"
            #[rustc_builtin_macro]
            macro_rules! stringify {() => {}}
            stringify!(a b c)
            "#,
        );

        assert_eq!(expanded, "\"a b c\"");
    }

    #[test]
    fn test_env_expand() {
        let expanded = expand_builtin_macro(
            r#"
            #[rustc_builtin_macro]
            macro_rules! env {() => {}}
            env!("TEST_ENV_VAR")
            "#,
        );

        assert_eq!(expanded, "\"\"");
    }

    #[test]
    fn test_option_env_expand() {
        let expanded = expand_builtin_macro(
            r#"
            #[rustc_builtin_macro]
            macro_rules! option_env {() => {}}
            option_env!("TEST_ENV_VAR")
            "#,
        );

        assert_eq!(expanded, "std::option::Option::None:: < &str>");
    }

    #[test]
    fn test_file_expand() {
        let expanded = expand_builtin_macro(
            r#"
            #[rustc_builtin_macro]
            macro_rules! file {() => {}}
            file!()
            "#,
        );

        assert_eq!(expanded, "\"\"");
    }

    #[test]
    fn test_compile_error_expand() {
        let expanded = expand_builtin_macro(
            r#"
            #[rustc_builtin_macro]
            macro_rules! compile_error {
                ($msg:expr) => ({ /* compiler built-in */ });
                ($msg:expr,) => ({ /* compiler built-in */ })
            }
            compile_error!("error!");
            "#,
        );

        assert_eq!(expanded, r#"loop{"error!"}"#);
    }

    #[test]
    fn test_format_args_expand() {
        let expanded = expand_builtin_macro(
            r#"
            #[rustc_builtin_macro]
            macro_rules! format_args {
                ($fmt:expr) => ({ /* compiler built-in */ });
                ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
            }
            format_args!("{} {:?}", arg1(a, b, c), arg2);
            "#,
        );

        assert_eq!(
            expanded,
            r#"std::fmt::Arguments::new_v1(&[], &[std::fmt::ArgumentV1::new(&(arg1(a,b,c)),std::fmt::Display::fmt),std::fmt::ArgumentV1::new(&(arg2),std::fmt::Display::fmt),])"#
        );
    }
}
