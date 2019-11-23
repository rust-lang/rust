//! Builtin macro
use crate::db::AstDatabase;
use crate::{
    ast::{self, AstNode},
    name, AstId, CrateId, HirFileId, MacroCallId, MacroDefId, MacroDefKind, MacroFileKind,
    TextUnit,
};

use crate::quote;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinFnLikeExpander {
    Column,
    File,
    Line,
    Stringify,
}

struct BuiltInMacroInfo {
    name: name::Name,
    kind: BuiltinFnLikeExpander,
    expand: fn(
        db: &dyn AstDatabase,
        id: MacroCallId,
        _tt: &tt::Subtree,
    ) -> Result<tt::Subtree, mbe::ExpandError>,
}

macro_rules! register_builtin {
    ( $(($name:ident, $kind: ident) => $expand:ident),* ) => {
        const BUILTIN_MACROS: &[BuiltInMacroInfo] = &[
            $(BuiltInMacroInfo { name: name::$name, kind: BuiltinFnLikeExpander::$kind, expand: $expand }),*
        ];
    };
}

register_builtin! {
    (COLUMN_MACRO, Column) => column_expand,
    (FILE_MACRO, File) => file_expand,
    (LINE_MACRO, Line) => line_expand,
    (STRINGIFY_MACRO, Stringify) => stringify_expand
}

impl BuiltinFnLikeExpander {
    pub fn expand(
        &self,
        db: &dyn AstDatabase,
        id: MacroCallId,
        tt: &tt::Subtree,
    ) -> Result<tt::Subtree, mbe::ExpandError> {
        let expander = BUILTIN_MACROS
            .iter()
            .find(|it| *self == it.kind)
            .map(|it| it.expand)
            .ok_or_else(|| mbe::ExpandError::ConversionError)?;

        expander(db, id, tt)
    }
}

pub fn find_builtin_macro(
    ident: &name::Name,
    krate: CrateId,
    ast_id: AstId<ast::MacroCall>,
) -> Option<MacroDefId> {
    let kind = BUILTIN_MACROS.iter().find(|it| *ident == it.name).map(|it| it.kind)?;

    Some(MacroDefId { krate, ast_id, kind: MacroDefKind::BuiltIn(kind) })
}

fn to_line_number(db: &dyn AstDatabase, file: HirFileId, pos: TextUnit) -> usize {
    // FIXME: Use expansion info
    let file_id = file.original_file(db);
    let text = db.file_text(file_id);
    let mut line_num = 1;

    // Count line end
    for (i, c) in text.chars().enumerate() {
        if i == pos.to_usize() {
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
    let macro_call = loc.ast_id.to_node(db);

    let arg = macro_call.token_tree().ok_or_else(|| mbe::ExpandError::UnexpectedToken)?;
    let arg_start = arg.syntax().text_range().start();

    let file = id.as_file(MacroFileKind::Expr);
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
    let macro_call = loc.ast_id.to_node(db);

    let macro_content = {
        let arg = macro_call.token_tree().ok_or_else(|| mbe::ExpandError::UnexpectedToken)?;
        let macro_args = arg.syntax().clone();
        let text = macro_args.text();
        let without_parens = TextUnit::of_char('(')..text.len() - TextUnit::of_char(')');
        text.slice(without_parens).to_string()
    };

    let expanded = quote! {
        #macro_content
    };

    Ok(expanded)
}

fn to_col_number(db: &dyn AstDatabase, file: HirFileId, pos: TextUnit) -> usize {
    // FIXME: Use expansion info
    let file_id = file.original_file(db);
    let text = db.file_text(file_id);
    let mut col_num = 1;

    for c in text[..pos.to_usize()].chars().rev() {
        if c == '\n' {
            break;
        }
        col_num = col_num + 1;
    }

    col_num
}

fn column_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    let loc = db.lookup_intern_macro(id);
    let macro_call = loc.ast_id.to_node(db);

    let _arg = macro_call.token_tree().ok_or_else(|| mbe::ExpandError::UnexpectedToken)?;
    let col_start = macro_call.syntax().text_range().start();

    let file = id.as_file(MacroFileKind::Expr);
    let col_num = to_col_number(db, file, col_start);

    let expanded = quote! {
        #col_num
    };

    Ok(expanded)
}

fn file_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
    _tt: &tt::Subtree,
) -> Result<tt::Subtree, mbe::ExpandError> {
    let loc = db.lookup_intern_macro(id);
    let macro_call = loc.ast_id.to_node(db);

    let _ = macro_call.token_tree().ok_or_else(|| mbe::ExpandError::UnexpectedToken)?;

    // FIXME: RA purposefully lacks knowledge of absolute file names
    // so just return "".
    let file_name = "";

    let expanded = quote! {
        #file_name
    };

    Ok(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_db::TestDB, MacroCallLoc};
    use ra_db::{fixture::WithFixture, SourceDatabase};

    fn expand_builtin_macro(s: &str, expander: BuiltinFnLikeExpander) -> String {
        let (db, file_id) = TestDB::with_single_file(&s);
        let parsed = db.parse(file_id);
        let macro_calls: Vec<_> =
            parsed.syntax_node().descendants().filter_map(|it| ast::MacroCall::cast(it)).collect();

        let ast_id_map = db.ast_id_map(file_id.into());

        // the first one should be a macro_rules
        let def = MacroDefId {
            krate: CrateId(0),
            ast_id: AstId::new(file_id.into(), ast_id_map.ast_id(&macro_calls[0])),
            kind: MacroDefKind::BuiltIn(expander),
        };

        let loc = MacroCallLoc {
            def,
            ast_id: AstId::new(file_id.into(), ast_id_map.ast_id(&macro_calls[1])),
        };

        let id = db.intern_macro(loc);
        let parsed = db.parse_or_expand(id.as_file(MacroFileKind::Expr)).unwrap();

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
            BuiltinFnLikeExpander::Column,
        );

        assert_eq!(expanded, "9");
    }

    #[test]
    fn test_line_expand() {
        let expanded = expand_builtin_macro(
            r#"
        #[rustc_builtin_macro]
        macro_rules! line {() => {}}
        line!()
"#,
            BuiltinFnLikeExpander::Line,
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
            BuiltinFnLikeExpander::Stringify,
        );

        assert_eq!(expanded, "\"a b c\"");
    }

    #[test]
    fn test_file_expand() {
        let expanded = expand_builtin_macro(
            r#"
        #[rustc_builtin_macro]
        macro_rules! file {() => {}}
        file!()
"#,
            BuiltinFnLikeExpander::File,
        );

        assert_eq!(expanded, "\"\"");
    }
}
