//! Builtin macro
use crate::db::AstDatabase;
use crate::{
    ast::{self, AstNode},
    name, AstId, CrateId, HirFileId, MacroCallId, MacroDefId, MacroDefKind, TextUnit,
};

use crate::quote;

macro_rules! register_builtin {
    ( $(($name:ident, $kind: ident) => $expand:ident),* ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinFnLikeExpander {
            $($kind),*
        }

        impl BuiltinFnLikeExpander {
            pub fn expand(
                &self,
                db: &dyn AstDatabase,
                id: MacroCallId,
                tt: &tt::Subtree,
            ) -> Result<tt::Subtree, mbe::ExpandError> {
                let expander = match *self {
                    $( BuiltinFnLikeExpander::$kind => $expand, )*
                };
                expander(db, id, tt)
            }

            fn by_name(ident: &name::Name) -> Option<BuiltinFnLikeExpander> {
                match ident {
                    $( id if id == &name::name![$name] => Some(BuiltinFnLikeExpander::$kind), )*
                    _ => return None,
                }
            }
        }

        pub fn find_builtin_macro(
            ident: &name::Name,
            krate: CrateId,
            ast_id: AstId<ast::MacroCall>,
        ) -> Option<MacroDefId> {
            let kind = BuiltinFnLikeExpander::by_name(ident)?;

            Some(MacroDefId { krate: Some(krate), ast_id: Some(ast_id), kind: MacroDefKind::BuiltIn(kind) })
        }
    };
}

register_builtin! {
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
    (format_args_nl, FormatArgsNl) => format_args_expand
}

fn to_line_number(db: &dyn AstDatabase, file: HirFileId, pos: TextUnit) -> usize {
    // FIXME: Use expansion info
    let file_id = file.original_file(db);
    let text = db.file_text(file_id);
    let mut line_num = 1;

    let pos = pos.to_usize();
    if pos > text.len() {
        // FIXME: `pos` at the moment could be an offset inside the "wrong" file
        // in this case, when we know it's wrong, we return a dummy value
        return 0;
    }
    // Count line end
    for (i, c) in text.chars().enumerate() {
        if i == pos {
            break;
        }
        if c == '\n' {
            line_num += 1;
        }
    }
    line_num
}

fn line_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    let loc = db.lookup_intern_macro(id);

    let arg = loc.kind.arg(db).ok_or_else(|| mbe::ExpandError::UnexpectedToken)?;
    let arg_start = arg.text_range().start();

    let file = id.as_file();
    let line_num = to_line_number(db, file, arg_start);

    let expanded = quote! {
        #line_num
    };

    Ok(expanded)
}

fn stringify_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
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
    _id: MacroCallId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    // dummy implementation for type-checking purposes
    let expanded = quote! { "" };

    Ok(expanded)
}

fn option_env_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    // dummy implementation for type-checking purposes
    let expanded = quote! { std::option::Option::None::<&str> };

    Ok(expanded)
}

fn to_col_number(db: &dyn AstDatabase, file: HirFileId, pos: TextUnit) -> usize {
    // FIXME: Use expansion info
    let file_id = file.original_file(db);
    let text = db.file_text(file_id);

    let pos = pos.to_usize();
    if pos > text.len() {
        // FIXME: `pos` at the moment could be an offset inside the "wrong" file
        // in this case we return a dummy value so that we don't `panic!`
        return 0;
    }

    let mut col_num = 1;
    for c in text[..pos].chars().rev() {
        if c == '\n' {
            break;
        }
        col_num += 1;
    }
    col_num
}

fn column_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    let loc = db.lookup_intern_macro(id);
    let macro_call = match loc.kind {
        crate::MacroCallKind::FnLike(ast_id) => ast_id.to_node(db),
        _ => panic!("column macro called as attr"),
    };

    let _arg = macro_call.token_tree().ok_or_else(|| mbe::ExpandError::UnexpectedToken)?;
    let col_start = macro_call.syntax().text_range().start();

    let file = id.as_file();
    let col_num = to_col_number(db, file, col_start);

    let expanded = quote! {
        #col_num
    };

    Ok(expanded)
}

fn file_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
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
    _id: MacroCallId,
    tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    if tt.count() == 1 {
        match &tt.token_trees[0] {
            tt::TokenTree::Leaf(tt::Leaf::Literal(it)) => {
                let s = it.text.as_str();
                if s.contains(r#"""#) {
                    return Ok(quote! { loop { #it }});
                }
            }
            _ => {}
        };
    }

    Err(mbe::ExpandError::BindingError("Must be a string".into()))
}

fn format_args_expand(
    _db: &dyn AstDatabase,
    _id: MacroCallId,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{name::AsName, test_db::TestDB, MacroCallKind, MacroCallLoc};
    use ra_db::{fixture::WithFixture, SourceDatabase};
    use ra_syntax::ast::NameOwner;

    fn expand_builtin_macro(s: &str) -> String {
        let (db, file_id) = TestDB::with_single_file(&s);
        let parsed = db.parse(file_id);
        let macro_calls: Vec<_> =
            parsed.syntax_node().descendants().filter_map(|it| ast::MacroCall::cast(it)).collect();

        let ast_id_map = db.ast_id_map(file_id.into());

        let expander =
            BuiltinFnLikeExpander::by_name(&macro_calls[0].name().unwrap().as_name()).unwrap();

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

        let id = db.intern_macro(loc);
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

        assert_eq!(expanded, "13");
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

        assert_eq!(expanded, "4");
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

        assert_eq!(expanded, "std::option::Option::None:: <&str>");
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
            r#"std::fmt::Arguments::new_v1(&[] ,&[std::fmt::ArgumentV1::new(&(arg1(a,b,c)),std::fmt::Display::fmt),std::fmt::ArgumentV1::new(&(arg2),std::fmt::Display::fmt),])"#
        );
    }
}
