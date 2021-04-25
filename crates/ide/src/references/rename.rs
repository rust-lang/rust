//! Renaming functionality
//!
//! All reference and file rename requests go through here where the corresponding [`SourceChange`]s
//! will be calculated.
use std::fmt::{self, Display};

use either::Either;
use hir::{AsAssocItem, InFile, Module, ModuleDef, ModuleSource, Semantics};
use ide_db::{
    base_db::{AnchoredPathBuf, FileId},
    defs::{Definition, NameClass, NameRefClass},
    search::FileReference,
    RootDatabase,
};
use stdx::never;
use syntax::{
    ast::{self, NameOwner},
    lex_single_syntax_kind, AstNode, SyntaxKind, SyntaxNode, T,
};

use text_edit::TextEdit;

use crate::{display::TryToNav, FilePosition, FileSystemEdit, RangeInfo, SourceChange, TextRange};

type RenameResult<T> = Result<T, RenameError>;
#[derive(Debug)]
pub struct RenameError(String);

impl fmt::Display for RenameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

macro_rules! format_err {
    ($fmt:expr) => {RenameError(format!($fmt))};
    ($fmt:expr, $($arg:tt)+) => {RenameError(format!($fmt, $($arg)+))}
}

macro_rules! bail {
    ($($tokens:tt)*) => {return Err(format_err!($($tokens)*))}
}

/// Prepares a rename. The sole job of this function is to return the TextRange of the thing that is
/// being targeted for a rename.
pub(crate) fn prepare_rename(
    db: &RootDatabase,
    position: FilePosition,
) -> RenameResult<RangeInfo<()>> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(position.file_id);
    let syntax = source_file.syntax();

    let def = find_definition(&sema, syntax, position)?;
    match def {
        Definition::SelfType(_) => bail!("Cannot rename `Self`"),
        Definition::ModuleDef(ModuleDef::BuiltinType(_)) => bail!("Cannot rename builtin type"),
        _ => {}
    };
    let nav =
        def.try_to_nav(sema.db).ok_or_else(|| format_err!("No references found at position"))?;
    nav.focus_range.ok_or_else(|| format_err!("No identifier available to rename"))?;

    let name_like = sema
        .find_node_at_offset_with_descend(&syntax, position.offset)
        .ok_or_else(|| format_err!("No references found at position"))?;
    let node = match &name_like {
        ast::NameLike::Name(it) => it.syntax(),
        ast::NameLike::NameRef(it) => it.syntax(),
        ast::NameLike::Lifetime(it) => it.syntax(),
    };
    Ok(RangeInfo::new(sema.original_range(node).range, ()))
}

// Feature: Rename
//
// Renames the item below the cursor and all of its references
//
// |===
// | Editor  | Shortcut
//
// | VS Code | kbd:[F2]
// |===
//
// image::https://user-images.githubusercontent.com/48062697/113065582-055aae80-91b1-11eb-8ade-2b58e6d81883.gif[]
pub(crate) fn rename(
    db: &RootDatabase,
    position: FilePosition,
    new_name: &str,
) -> RenameResult<SourceChange> {
    let sema = Semantics::new(db);
    rename_with_semantics(&sema, position, new_name)
}

pub(crate) fn rename_with_semantics(
    sema: &Semantics<RootDatabase>,
    position: FilePosition,
    new_name: &str,
) -> RenameResult<SourceChange> {
    let source_file = sema.parse(position.file_id);
    let syntax = source_file.syntax();

    let def = find_definition(sema, syntax, position)?;
    match def {
        Definition::ModuleDef(ModuleDef::Module(module)) => rename_mod(&sema, module, new_name),
        Definition::SelfType(_) => bail!("Cannot rename `Self`"),
        Definition::ModuleDef(ModuleDef::BuiltinType(_)) => bail!("Cannot rename builtin type"),
        def => rename_reference(sema, def, new_name),
    }
}

/// Called by the client when it is about to rename a file.
pub(crate) fn will_rename_file(
    db: &RootDatabase,
    file_id: FileId,
    new_name_stem: &str,
) -> Option<SourceChange> {
    let sema = Semantics::new(db);
    let module = sema.to_module_def(file_id)?;
    let mut change = rename_mod(&sema, module, new_name_stem).ok()?;
    change.file_system_edits.clear();
    Some(change)
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum IdentifierKind {
    Ident,
    Lifetime,
    ToSelf,
    Underscore,
}

fn check_identifier(new_name: &str) -> RenameResult<IdentifierKind> {
    match lex_single_syntax_kind(new_name) {
        Some(res) => match res {
            (SyntaxKind::IDENT, _) => Ok(IdentifierKind::Ident),
            (T![_], _) => Ok(IdentifierKind::Underscore),
            (T![self], _) => Ok(IdentifierKind::ToSelf),
            (SyntaxKind::LIFETIME_IDENT, _) if new_name != "'static" && new_name != "'_" => {
                Ok(IdentifierKind::Lifetime)
            }
            (SyntaxKind::LIFETIME_IDENT, _) => {
                bail!("Invalid name `{}`: not a lifetime identifier", new_name)
            }
            (_, Some(syntax_error)) => bail!("Invalid name `{}`: {}", new_name, syntax_error),
            (_, None) => bail!("Invalid name `{}`: not an identifier", new_name),
        },
        None => bail!("Invalid name `{}`: not an identifier", new_name),
    }
}

fn find_definition(
    sema: &Semantics<RootDatabase>,
    syntax: &SyntaxNode,
    position: FilePosition,
) -> RenameResult<Definition> {
    match sema
        .find_node_at_offset_with_descend(syntax, position.offset)
        .ok_or_else(|| format_err!("No references found at position"))?
    {
        // renaming aliases would rename the item being aliased as the HIR doesn't track aliases yet
        ast::NameLike::Name(name)
            if name.syntax().parent().map_or(false, |it| ast::Rename::can_cast(it.kind())) =>
        {
            bail!("Renaming aliases is currently unsupported")
        }
        ast::NameLike::Name(name) => {
            NameClass::classify(sema, &name).map(|class| class.referenced_or_defined(sema.db))
        }
        ast::NameLike::NameRef(name_ref) => {
            NameRefClass::classify(sema, &name_ref).map(|class| class.referenced(sema.db))
        }
        ast::NameLike::Lifetime(lifetime) => NameRefClass::classify_lifetime(sema, &lifetime)
            .map(|class| NameRefClass::referenced(class, sema.db))
            .or_else(|| {
                NameClass::classify_lifetime(sema, &lifetime)
                    .map(|it| it.referenced_or_defined(sema.db))
            }),
    }
    .ok_or_else(|| format_err!("No references found at position"))
}

fn rename_mod(
    sema: &Semantics<RootDatabase>,
    module: Module,
    new_name: &str,
) -> RenameResult<SourceChange> {
    if IdentifierKind::Ident != check_identifier(new_name)? {
        bail!("Invalid name `{0}`: cannot rename module to {0}", new_name);
    }

    let mut source_change = SourceChange::default();

    let InFile { file_id, value: def_source } = module.definition_source(sema.db);
    let file_id = file_id.original_file(sema.db);
    if let ModuleSource::SourceFile(..) = def_source {
        // mod is defined in path/to/dir/mod.rs
        let path = if module.is_mod_rs(sema.db) {
            format!("../{}/mod.rs", new_name)
        } else {
            format!("{}.rs", new_name)
        };
        let dst = AnchoredPathBuf { anchor: file_id, path };
        let move_file = FileSystemEdit::MoveFile { src: file_id, dst };
        source_change.push_file_system_edit(move_file);
    }

    if let Some(InFile { file_id, value: decl_source }) = module.declaration_source(sema.db) {
        let file_id = file_id.original_file(sema.db);
        match decl_source.name() {
            Some(name) => source_change.insert_source_edit(
                file_id,
                TextEdit::replace(name.syntax().text_range(), new_name.to_string()),
            ),
            _ => never!("Module source node is missing a name"),
        }
    }
    let def = Definition::ModuleDef(ModuleDef::Module(module));
    let usages = def.usages(sema).all();
    let ref_edits = usages.iter().map(|(&file_id, references)| {
        (file_id, source_edit_from_references(references, def, new_name))
    });
    source_change.extend(ref_edits);

    Ok(source_change)
}

fn rename_reference(
    sema: &Semantics<RootDatabase>,
    def: Definition,
    new_name: &str,
) -> RenameResult<SourceChange> {
    let ident_kind = check_identifier(new_name)?;

    if matches!(
        def, // is target a lifetime?
        Definition::GenericParam(hir::GenericParam::LifetimeParam(_)) | Definition::Label(_)
    ) {
        match ident_kind {
            IdentifierKind::Ident | IdentifierKind::ToSelf | IdentifierKind::Underscore => {
                cov_mark::hit!(rename_not_a_lifetime_ident_ref);
                bail!("Invalid name `{}`: not a lifetime identifier", new_name);
            }
            IdentifierKind::Lifetime => cov_mark::hit!(rename_lifetime),
        }
    } else {
        match (ident_kind, def) {
            (IdentifierKind::Lifetime, _) => {
                cov_mark::hit!(rename_not_an_ident_ref);
                bail!("Invalid name `{}`: not an identifier", new_name);
            }
            (IdentifierKind::ToSelf, Definition::Local(local)) => {
                if local.is_self(sema.db) {
                    // no-op
                    cov_mark::hit!(rename_self_to_self);
                    return Ok(SourceChange::default());
                } else {
                    cov_mark::hit!(rename_to_self);
                    return rename_to_self(sema, local);
                }
            }
            (ident_kind, Definition::Local(local)) => {
                if let Some(self_param) = local.as_self_param(sema.db) {
                    cov_mark::hit!(rename_self_to_param);
                    return rename_self_to_param(sema, local, self_param, new_name, ident_kind);
                } else {
                    cov_mark::hit!(rename_local);
                }
            }
            (IdentifierKind::ToSelf, _) => bail!("Invalid name `{}`: not an identifier", new_name),
            (IdentifierKind::Ident, _) => cov_mark::hit!(rename_non_local),
            (IdentifierKind::Underscore, _) => (),
        }
    }

    let usages = def.usages(sema).all();
    if !usages.is_empty() && ident_kind == IdentifierKind::Underscore {
        cov_mark::hit!(rename_underscore_multiple);
        bail!("Cannot rename reference to `_` as it is being referenced multiple times");
    }
    let mut source_change = SourceChange::default();
    source_change.extend(usages.iter().map(|(&file_id, references)| {
        (file_id, source_edit_from_references(&references, def, new_name))
    }));

    let (file_id, edit) = source_edit_from_def(sema, def, new_name)?;
    source_change.insert_source_edit(file_id, edit);
    Ok(source_change)
}

fn rename_to_self(sema: &Semantics<RootDatabase>, local: hir::Local) -> RenameResult<SourceChange> {
    if never!(local.is_self(sema.db)) {
        bail!("rename_to_self invoked on self");
    }

    let fn_def = match local.parent(sema.db) {
        hir::DefWithBody::Function(func) => func,
        _ => bail!("Cannot rename local to self outside of function"),
    };

    if let Some(_) = fn_def.self_param(sema.db) {
        bail!("Method already has a self parameter");
    }

    let params = fn_def.assoc_fn_params(sema.db);
    let first_param = params
        .first()
        .ok_or_else(|| format_err!("Cannot rename local to self unless it is a parameter"))?;
    if first_param.as_local(sema.db) != local {
        bail!("Only the first parameter may be renamed to self");
    }

    let assoc_item = fn_def
        .as_assoc_item(sema.db)
        .ok_or_else(|| format_err!("Cannot rename parameter to self for free function"))?;
    let impl_ = match assoc_item.container(sema.db) {
        hir::AssocItemContainer::Trait(_) => {
            bail!("Cannot rename parameter to self for trait functions");
        }
        hir::AssocItemContainer::Impl(impl_) => impl_,
    };
    let first_param_ty = first_param.ty();
    let impl_ty = impl_.self_ty(sema.db);
    let (ty, self_param) = if impl_ty.remove_ref().is_some() {
        // if the impl is a ref to the type we can just match the `&T` with self directly
        (first_param_ty.clone(), "self")
    } else {
        first_param_ty.remove_ref().map_or((first_param_ty.clone(), "self"), |ty| {
            (ty, if first_param_ty.is_mutable_reference() { "&mut self" } else { "&self" })
        })
    };

    if ty != impl_ty {
        bail!("Parameter type differs from impl block type");
    }

    let InFile { file_id, value: param_source } =
        first_param.source(sema.db).ok_or_else(|| format_err!("No source for parameter found"))?;

    let def = Definition::Local(local);
    let usages = def.usages(sema).all();
    let mut source_change = SourceChange::default();
    source_change.extend(usages.iter().map(|(&file_id, references)| {
        (file_id, source_edit_from_references(references, def, "self"))
    }));
    source_change.insert_source_edit(
        file_id.original_file(sema.db),
        TextEdit::replace(param_source.syntax().text_range(), String::from(self_param)),
    );
    Ok(source_change)
}

fn rename_self_to_param(
    sema: &Semantics<RootDatabase>,
    local: hir::Local,
    self_param: hir::SelfParam,
    new_name: &str,
    identifier_kind: IdentifierKind,
) -> RenameResult<SourceChange> {
    let InFile { file_id, value: self_param } =
        self_param.source(sema.db).ok_or_else(|| format_err!("cannot find function source"))?;

    let def = Definition::Local(local);
    let usages = def.usages(sema).all();
    let edit = text_edit_from_self_param(&self_param, new_name)
        .ok_or_else(|| format_err!("No target type found"))?;
    if usages.len() > 1 && identifier_kind == IdentifierKind::Underscore {
        bail!("Cannot rename reference to `_` as it is being referenced multiple times");
    }
    let mut source_change = SourceChange::default();
    source_change.insert_source_edit(file_id.original_file(sema.db), edit);
    source_change.extend(usages.iter().map(|(&file_id, references)| {
        (file_id, source_edit_from_references(&references, def, new_name))
    }));
    Ok(source_change)
}

fn text_edit_from_self_param(self_param: &ast::SelfParam, new_name: &str) -> Option<TextEdit> {
    fn target_type_name(impl_def: &ast::Impl) -> Option<String> {
        if let Some(ast::Type::PathType(p)) = impl_def.self_ty() {
            return Some(p.path()?.segment()?.name_ref()?.text().to_string());
        }
        None
    }

    let impl_def = self_param.syntax().ancestors().find_map(|it| ast::Impl::cast(it))?;
    let type_name = target_type_name(&impl_def)?;

    let mut replacement_text = String::from(new_name);
    replacement_text.push_str(": ");
    match (self_param.amp_token(), self_param.mut_token()) {
        (Some(_), None) => replacement_text.push('&'),
        (Some(_), Some(_)) => replacement_text.push_str("&mut "),
        (_, _) => (),
    };
    replacement_text.push_str(type_name.as_str());

    Some(TextEdit::replace(self_param.syntax().text_range(), replacement_text))
}

fn source_edit_from_references(
    references: &[FileReference],
    def: Definition,
    new_name: &str,
) -> TextEdit {
    let mut edit = TextEdit::builder();
    for reference in references {
        let (range, replacement) = match &reference.name {
            // if the ranges differ then the node is inside a macro call, we can't really attempt
            // to make special rewrites like shorthand syntax and such, so just rename the node in
            // the macro input
            ast::NameLike::NameRef(name_ref)
                if name_ref.syntax().text_range() == reference.range =>
            {
                source_edit_from_name_ref(name_ref, new_name, def)
            }
            ast::NameLike::Name(name) if name.syntax().text_range() == reference.range => {
                source_edit_from_name(name, new_name)
            }
            _ => None,
        }
        .unwrap_or_else(|| (reference.range, new_name.to_string()));
        edit.replace(range, replacement);
    }
    edit.finish()
}

fn source_edit_from_name(name: &ast::Name, new_name: &str) -> Option<(TextRange, String)> {
    if let Some(_) = ast::RecordPatField::for_field_name(name) {
        if let Some(ident_pat) = name.syntax().parent().and_then(ast::IdentPat::cast) {
            return Some((
                TextRange::empty(ident_pat.syntax().text_range().start()),
                [new_name, ": "].concat(),
            ));
        }
    }
    None
}

fn source_edit_from_name_ref(
    name_ref: &ast::NameRef,
    new_name: &str,
    def: Definition,
) -> Option<(TextRange, String)> {
    if let Some(record_field) = ast::RecordExprField::for_name_ref(name_ref) {
        let rcf_name_ref = record_field.name_ref();
        let rcf_expr = record_field.expr();
        match (rcf_name_ref, rcf_expr.and_then(|it| it.name_ref())) {
            // field: init-expr, check if we can use a field init shorthand
            (Some(field_name), Some(init)) => {
                if field_name == *name_ref {
                    if init.text() == new_name {
                        cov_mark::hit!(test_rename_field_put_init_shorthand);
                        // same names, we can use a shorthand here instead.
                        // we do not want to erase attributes hence this range start
                        let s = field_name.syntax().text_range().start();
                        let e = record_field.syntax().text_range().end();
                        return Some((TextRange::new(s, e), new_name.to_owned()));
                    }
                } else if init == *name_ref {
                    if field_name.text() == new_name {
                        cov_mark::hit!(test_rename_local_put_init_shorthand);
                        // same names, we can use a shorthand here instead.
                        // we do not want to erase attributes hence this range start
                        let s = field_name.syntax().text_range().start();
                        let e = record_field.syntax().text_range().end();
                        return Some((TextRange::new(s, e), new_name.to_owned()));
                    }
                }
                None
            }
            // init shorthand
            // FIXME: instead of splitting the shorthand, recursively trigger a rename of the
            // other name https://github.com/rust-analyzer/rust-analyzer/issues/6547
            (None, Some(_)) if matches!(def, Definition::Field(_)) => {
                cov_mark::hit!(test_rename_field_in_field_shorthand);
                let s = name_ref.syntax().text_range().start();
                Some((TextRange::empty(s), format!("{}: ", new_name)))
            }
            (None, Some(_)) if matches!(def, Definition::Local(_)) => {
                cov_mark::hit!(test_rename_local_in_field_shorthand);
                let s = name_ref.syntax().text_range().end();
                Some((TextRange::empty(s), format!(": {}", new_name)))
            }
            _ => None,
        }
    } else if let Some(record_field) = ast::RecordPatField::for_field_name_ref(name_ref) {
        let rcf_name_ref = record_field.name_ref();
        let rcf_pat = record_field.pat();
        match (rcf_name_ref, rcf_pat) {
            // field: rename
            (Some(field_name), Some(ast::Pat::IdentPat(pat))) if field_name == *name_ref => {
                // field name is being renamed
                if pat.name().map_or(false, |it| it.text() == new_name) {
                    cov_mark::hit!(test_rename_field_put_init_shorthand_pat);
                    // same names, we can use a shorthand here instead/
                    // we do not want to erase attributes hence this range start
                    let s = field_name.syntax().text_range().start();
                    let e = record_field.syntax().text_range().end();
                    Some((TextRange::new(s, e), pat.to_string()))
                } else {
                    None
                }
            }
            _ => None,
        }
    } else {
        None
    }
}

fn source_edit_from_def(
    sema: &Semantics<RootDatabase>,
    def: Definition,
    new_name: &str,
) -> RenameResult<(FileId, TextEdit)> {
    let nav =
        def.try_to_nav(sema.db).ok_or_else(|| format_err!("No references found at position"))?;

    let mut replacement_text = String::new();
    let mut repl_range =
        nav.focus_range.ok_or_else(|| format_err!("No identifier available to rename"))?;
    if let Definition::Local(local) = def {
        if let Either::Left(pat) = local.source(sema.db).value {
            if matches!(
                pat.syntax().parent().and_then(ast::RecordPatField::cast),
                Some(pat_field) if pat_field.name_ref().is_none()
            ) {
                replacement_text.push_str(": ");
                replacement_text.push_str(new_name);
                repl_range = TextRange::new(
                    pat.syntax().text_range().end(),
                    pat.syntax().text_range().end(),
                );
            }
        }
    }
    if replacement_text.is_empty() {
        replacement_text.push_str(new_name);
    }
    let edit = TextEdit::replace(repl_range, replacement_text);
    Ok((nav.file_id, edit))
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use stdx::trim_indent;
    use test_utils::assert_eq_text;
    use text_edit::TextEdit;

    use crate::{fixture, FileId};

    use super::{RangeInfo, RenameError};

    fn check(new_name: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
        let ra_fixture_after = &trim_indent(ra_fixture_after);
        let (analysis, position) = fixture::position(ra_fixture_before);
        let rename_result = analysis
            .rename(position, new_name)
            .unwrap_or_else(|err| panic!("Rename to '{}' was cancelled: {}", new_name, err));
        match rename_result {
            Ok(source_change) => {
                let mut text_edit_builder = TextEdit::builder();
                let mut file_id: Option<FileId> = None;
                for edit in source_change.source_file_edits {
                    file_id = Some(edit.0);
                    for indel in edit.1.into_iter() {
                        text_edit_builder.replace(indel.delete, indel.insert);
                    }
                }
                if let Some(file_id) = file_id {
                    let mut result = analysis.file_text(file_id).unwrap().to_string();
                    text_edit_builder.finish().apply(&mut result);
                    assert_eq_text!(ra_fixture_after, &*result);
                }
            }
            Err(err) => {
                if ra_fixture_after.starts_with("error:") {
                    let error_message = ra_fixture_after
                        .chars()
                        .into_iter()
                        .skip("error:".len())
                        .collect::<String>();
                    assert_eq!(error_message.trim(), err.to_string());
                    return;
                } else {
                    panic!("Rename to '{}' failed unexpectedly: {}", new_name, err)
                }
            }
        };
    }

    fn check_expect(new_name: &str, ra_fixture: &str, expect: Expect) {
        let (analysis, position) = fixture::position(ra_fixture);
        let source_change =
            analysis.rename(position, new_name).unwrap().expect("Expect returned a RenameError");
        expect.assert_debug_eq(&source_change)
    }

    fn check_prepare(ra_fixture: &str, expect: Expect) {
        let (analysis, position) = fixture::position(ra_fixture);
        let result = analysis
            .prepare_rename(position)
            .unwrap_or_else(|err| panic!("PrepareRename was cancelled: {}", err));
        match result {
            Ok(RangeInfo { range, info: () }) => {
                let source = analysis.file_text(position.file_id).unwrap();
                expect.assert_eq(&format!("{:?}: {}", range, &source[range]))
            }
            Err(RenameError(err)) => expect.assert_eq(&err),
        };
    }

    #[test]
    fn test_prepare_rename_namelikes() {
        check_prepare(r"fn name$0<'lifetime>() {}", expect![[r#"3..7: name"#]]);
        check_prepare(r"fn name<'lifetime$0>() {}", expect![[r#"8..17: 'lifetime"#]]);
        check_prepare(r"fn name<'lifetime>() { name$0(); }", expect![[r#"23..27: name"#]]);
    }

    #[test]
    fn test_prepare_rename_in_macro() {
        check_prepare(
            r"macro_rules! foo {
    ($ident:ident) => {
        pub struct $ident;
    }
}
foo!(Foo$0);",
            expect![[r#"83..86: Foo"#]],
        );
    }

    #[test]
    fn test_prepare_rename_keyword() {
        check_prepare(r"struct$0 Foo;", expect![[r#"No references found at position"#]]);
    }

    #[test]
    fn test_prepare_rename_tuple_field() {
        check_prepare(
            r#"
struct Foo(i32);

fn baz() {
    let mut x = Foo(4);
    x.0$0 = 5;
}
"#,
            expect![[r#"No identifier available to rename"#]],
        );
    }

    #[test]
    fn test_prepare_rename_builtin() {
        check_prepare(
            r#"
fn foo() {
    let x: i32$0 = 0;
}
"#,
            expect![[r#"Cannot rename builtin type"#]],
        );
    }

    #[test]
    fn test_prepare_rename_self() {
        check_prepare(
            r#"
struct Foo {}

impl Foo {
    fn foo(self) -> Self$0 {
        self
    }
}
"#,
            expect![[r#"Cannot rename `Self`"#]],
        );
    }

    #[test]
    fn test_rename_to_underscore() {
        check("_", r#"fn main() { let i$0 = 1; }"#, r#"fn main() { let _ = 1; }"#);
    }

    #[test]
    fn test_rename_to_raw_identifier() {
        check("r#fn", r#"fn main() { let i$0 = 1; }"#, r#"fn main() { let r#fn = 1; }"#);
    }

    #[test]
    fn test_rename_to_invalid_identifier1() {
        check(
            "invalid!",
            r#"fn main() { let i$0 = 1; }"#,
            "error: Invalid name `invalid!`: not an identifier",
        );
    }

    #[test]
    fn test_rename_to_invalid_identifier2() {
        check(
            "multiple tokens",
            r#"fn main() { let i$0 = 1; }"#,
            "error: Invalid name `multiple tokens`: not an identifier",
        );
    }

    #[test]
    fn test_rename_to_invalid_identifier3() {
        check(
            "let",
            r#"fn main() { let i$0 = 1; }"#,
            "error: Invalid name `let`: not an identifier",
        );
    }

    #[test]
    fn test_rename_to_invalid_identifier_lifetime() {
        cov_mark::check!(rename_not_an_ident_ref);
        check(
            "'foo",
            r#"fn main() { let i$0 = 1; }"#,
            "error: Invalid name `'foo`: not an identifier",
        );
    }

    #[test]
    fn test_rename_to_invalid_identifier_lifetime2() {
        cov_mark::check!(rename_not_a_lifetime_ident_ref);
        check(
            "foo",
            r#"fn main<'a>(_: &'a$0 ()) {}"#,
            "error: Invalid name `foo`: not a lifetime identifier",
        );
    }

    #[test]
    fn test_rename_to_underscore_invalid() {
        cov_mark::check!(rename_underscore_multiple);
        check(
            "_",
            r#"fn main(foo$0: ()) {foo;}"#,
            "error: Cannot rename reference to `_` as it is being referenced multiple times",
        );
    }

    #[test]
    fn test_rename_mod_invalid() {
        check(
            "'foo",
            r#"mod foo$0 {}"#,
            "error: Invalid name `'foo`: cannot rename module to 'foo",
        );
    }

    #[test]
    fn test_rename_for_local() {
        cov_mark::check!(rename_local);
        check(
            "k",
            r#"
fn main() {
    let mut i = 1;
    let j = 1;
    i = i$0 + j;

    { i = 0; }

    i = 5;
}
"#,
            r#"
fn main() {
    let mut k = 1;
    let j = 1;
    k = k + j;

    { k = 0; }

    k = 5;
}
"#,
        );
    }

    #[test]
    fn test_rename_unresolved_reference() {
        check(
            "new_name",
            r#"fn main() { let _ = unresolved_ref$0; }"#,
            "error: No references found at position",
        );
    }

    #[test]
    fn test_rename_for_macro_args() {
        check(
            "b",
            r#"
macro_rules! foo {($i:ident) => {$i} }
fn main() {
    let a$0 = "test";
    foo!(a);
}
"#,
            r#"
macro_rules! foo {($i:ident) => {$i} }
fn main() {
    let b = "test";
    foo!(b);
}
"#,
        );
    }

    #[test]
    fn test_rename_for_macro_args_rev() {
        check(
            "b",
            r#"
macro_rules! foo {($i:ident) => {$i} }
fn main() {
    let a = "test";
    foo!(a$0);
}
"#,
            r#"
macro_rules! foo {($i:ident) => {$i} }
fn main() {
    let b = "test";
    foo!(b);
}
"#,
        );
    }

    #[test]
    fn test_rename_for_macro_define_fn() {
        check(
            "bar",
            r#"
macro_rules! define_fn {($id:ident) => { fn $id{} }}
define_fn!(foo);
fn main() {
    fo$0o();
}
"#,
            r#"
macro_rules! define_fn {($id:ident) => { fn $id{} }}
define_fn!(bar);
fn main() {
    bar();
}
"#,
        );
    }

    #[test]
    fn test_rename_for_macro_define_fn_rev() {
        check(
            "bar",
            r#"
macro_rules! define_fn {($id:ident) => { fn $id{} }}
define_fn!(fo$0o);
fn main() {
    foo();
}
"#,
            r#"
macro_rules! define_fn {($id:ident) => { fn $id{} }}
define_fn!(bar);
fn main() {
    bar();
}
"#,
        );
    }

    #[test]
    fn test_rename_for_param_inside() {
        check("j", r#"fn foo(i : u32) -> u32 { i$0 }"#, r#"fn foo(j : u32) -> u32 { j }"#);
    }

    #[test]
    fn test_rename_refs_for_fn_param() {
        check("j", r#"fn foo(i$0 : u32) -> u32 { i }"#, r#"fn foo(j : u32) -> u32 { j }"#);
    }

    #[test]
    fn test_rename_for_mut_param() {
        check("j", r#"fn foo(mut i$0 : u32) -> u32 { i }"#, r#"fn foo(mut j : u32) -> u32 { j }"#);
    }

    #[test]
    fn test_rename_struct_field() {
        check(
            "j",
            r#"
struct Foo { i$0: i32 }

impl Foo {
    fn new(i: i32) -> Self {
        Self { i: i }
    }
}
"#,
            r#"
struct Foo { j: i32 }

impl Foo {
    fn new(i: i32) -> Self {
        Self { j: i }
    }
}
"#,
        );
    }

    #[test]
    fn test_rename_field_in_field_shorthand() {
        cov_mark::check!(test_rename_field_in_field_shorthand);
        check(
            "j",
            r#"
struct Foo { i$0: i32 }

impl Foo {
    fn new(i: i32) -> Self {
        Self { i }
    }
}
"#,
            r#"
struct Foo { j: i32 }

impl Foo {
    fn new(i: i32) -> Self {
        Self { j: i }
    }
}
"#,
        );
    }

    #[test]
    fn test_rename_local_in_field_shorthand() {
        cov_mark::check!(test_rename_local_in_field_shorthand);
        check(
            "j",
            r#"
struct Foo { i: i32 }

impl Foo {
    fn new(i$0: i32) -> Self {
        Self { i }
    }
}
"#,
            r#"
struct Foo { i: i32 }

impl Foo {
    fn new(j: i32) -> Self {
        Self { i: j }
    }
}
"#,
        );
    }

    #[test]
    fn test_field_shorthand_correct_struct() {
        check(
            "j",
            r#"
struct Foo { i$0: i32 }
struct Bar { i: i32 }

impl Bar {
    fn new(i: i32) -> Self {
        Self { i }
    }
}
"#,
            r#"
struct Foo { j: i32 }
struct Bar { i: i32 }

impl Bar {
    fn new(i: i32) -> Self {
        Self { i }
    }
}
"#,
        );
    }

    #[test]
    fn test_shadow_local_for_struct_shorthand() {
        check(
            "j",
            r#"
struct Foo { i: i32 }

fn baz(i$0: i32) -> Self {
     let x = Foo { i };
     {
         let i = 0;
         Foo { i }
     }
}
"#,
            r#"
struct Foo { i: i32 }

fn baz(j: i32) -> Self {
     let x = Foo { i: j };
     {
         let i = 0;
         Foo { i }
     }
}
"#,
        );
    }

    #[test]
    fn test_rename_mod() {
        check_expect(
            "foo2",
            r#"
//- /lib.rs
mod bar;

//- /bar.rs
mod foo$0;

//- /bar/foo.rs
// empty
"#,
            expect![[r#"
                SourceChange {
                    source_file_edits: {
                        FileId(
                            1,
                        ): TextEdit {
                            indels: [
                                Indel {
                                    insert: "foo2",
                                    delete: 4..7,
                                },
                            ],
                        },
                    },
                    file_system_edits: [
                        MoveFile {
                            src: FileId(
                                2,
                            ),
                            dst: AnchoredPathBuf {
                                anchor: FileId(
                                    2,
                                ),
                                path: "foo2.rs",
                            },
                        },
                    ],
                    is_snippet: false,
                }
            "#]],
        );
    }

    #[test]
    fn test_rename_mod_in_use_tree() {
        check_expect(
            "quux",
            r#"
//- /main.rs
pub mod foo;
pub mod bar;
fn main() {}

//- /foo.rs
pub struct FooContent;

//- /bar.rs
use crate::foo$0::FooContent;
"#,
            expect![[r#"
                SourceChange {
                    source_file_edits: {
                        FileId(
                            0,
                        ): TextEdit {
                            indels: [
                                Indel {
                                    insert: "quux",
                                    delete: 8..11,
                                },
                            ],
                        },
                        FileId(
                            2,
                        ): TextEdit {
                            indels: [
                                Indel {
                                    insert: "quux",
                                    delete: 11..14,
                                },
                            ],
                        },
                    },
                    file_system_edits: [
                        MoveFile {
                            src: FileId(
                                1,
                            ),
                            dst: AnchoredPathBuf {
                                anchor: FileId(
                                    1,
                                ),
                                path: "quux.rs",
                            },
                        },
                    ],
                    is_snippet: false,
                }
            "#]],
        );
    }

    #[test]
    fn test_rename_mod_in_dir() {
        check_expect(
            "foo2",
            r#"
//- /lib.rs
mod fo$0o;
//- /foo/mod.rs
// empty
"#,
            expect![[r#"
                SourceChange {
                    source_file_edits: {
                        FileId(
                            0,
                        ): TextEdit {
                            indels: [
                                Indel {
                                    insert: "foo2",
                                    delete: 4..7,
                                },
                            ],
                        },
                    },
                    file_system_edits: [
                        MoveFile {
                            src: FileId(
                                1,
                            ),
                            dst: AnchoredPathBuf {
                                anchor: FileId(
                                    1,
                                ),
                                path: "../foo2/mod.rs",
                            },
                        },
                    ],
                    is_snippet: false,
                }
            "#]],
        );
    }

    #[test]
    fn test_rename_unusually_nested_mod() {
        check_expect(
            "bar",
            r#"
//- /lib.rs
mod outer { mod fo$0o; }

//- /outer/foo.rs
// empty
"#,
            expect![[r#"
                SourceChange {
                    source_file_edits: {
                        FileId(
                            0,
                        ): TextEdit {
                            indels: [
                                Indel {
                                    insert: "bar",
                                    delete: 16..19,
                                },
                            ],
                        },
                    },
                    file_system_edits: [
                        MoveFile {
                            src: FileId(
                                1,
                            ),
                            dst: AnchoredPathBuf {
                                anchor: FileId(
                                    1,
                                ),
                                path: "bar.rs",
                            },
                        },
                    ],
                    is_snippet: false,
                }
            "#]],
        );
    }

    #[test]
    fn test_module_rename_in_path() {
        check(
            "baz",
            r#"
mod $0foo { pub fn bar() {} }

fn main() { foo::bar(); }
"#,
            r#"
mod baz { pub fn bar() {} }

fn main() { baz::bar(); }
"#,
        );
    }

    #[test]
    fn test_rename_mod_filename_and_path() {
        check_expect(
            "foo2",
            r#"
//- /lib.rs
mod bar;
fn f() {
    bar::foo::fun()
}

//- /bar.rs
pub mod foo$0;

//- /bar/foo.rs
// pub fn fun() {}
"#,
            expect![[r#"
                SourceChange {
                    source_file_edits: {
                        FileId(
                            0,
                        ): TextEdit {
                            indels: [
                                Indel {
                                    insert: "foo2",
                                    delete: 27..30,
                                },
                            ],
                        },
                        FileId(
                            1,
                        ): TextEdit {
                            indels: [
                                Indel {
                                    insert: "foo2",
                                    delete: 8..11,
                                },
                            ],
                        },
                    },
                    file_system_edits: [
                        MoveFile {
                            src: FileId(
                                2,
                            ),
                            dst: AnchoredPathBuf {
                                anchor: FileId(
                                    2,
                                ),
                                path: "foo2.rs",
                            },
                        },
                    ],
                    is_snippet: false,
                }
            "#]],
        );
    }

    #[test]
    fn test_enum_variant_from_module_1() {
        cov_mark::check!(rename_non_local);
        check(
            "Baz",
            r#"
mod foo {
    pub enum Foo { Bar$0 }
}

fn func(f: foo::Foo) {
    match f {
        foo::Foo::Bar => {}
    }
}
"#,
            r#"
mod foo {
    pub enum Foo { Baz }
}

fn func(f: foo::Foo) {
    match f {
        foo::Foo::Baz => {}
    }
}
"#,
        );
    }

    #[test]
    fn test_enum_variant_from_module_2() {
        check(
            "baz",
            r#"
mod foo {
    pub struct Foo { pub bar$0: uint }
}

fn foo(f: foo::Foo) {
    let _ = f.bar;
}
"#,
            r#"
mod foo {
    pub struct Foo { pub baz: uint }
}

fn foo(f: foo::Foo) {
    let _ = f.baz;
}
"#,
        );
    }

    #[test]
    fn test_parameter_to_self() {
        cov_mark::check!(rename_to_self);
        check(
            "self",
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(foo$0: &mut Foo) -> i32 {
        foo.i
    }
}
"#,
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(&mut self) -> i32 {
        self.i
    }
}
"#,
        );
        check(
            "self",
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(foo$0: Foo) -> i32 {
        foo.i
    }
}
"#,
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(self) -> i32 {
        self.i
    }
}
"#,
        );
    }

    #[test]
    fn test_parameter_to_self_error_no_impl() {
        check(
            "self",
            r#"
struct Foo { i: i32 }

fn f(foo$0: &mut Foo) -> i32 {
    foo.i
}
"#,
            "error: Cannot rename parameter to self for free function",
        );
        check(
            "self",
            r#"
struct Foo { i: i32 }
struct Bar;

impl Bar {
    fn f(foo$0: &mut Foo) -> i32 {
        foo.i
    }
}
"#,
            "error: Parameter type differs from impl block type",
        );
    }

    #[test]
    fn test_parameter_to_self_error_not_first() {
        check(
            "self",
            r#"
struct Foo { i: i32 }
impl Foo {
    fn f(x: (), foo$0: &mut Foo) -> i32 {
        foo.i
    }
}
"#,
            "error: Only the first parameter may be renamed to self",
        );
    }

    #[test]
    fn test_parameter_to_self_impl_ref() {
        check(
            "self",
            r#"
struct Foo { i: i32 }
impl &Foo {
    fn f(foo$0: &Foo) -> i32 {
        foo.i
    }
}
"#,
            r#"
struct Foo { i: i32 }
impl &Foo {
    fn f(self) -> i32 {
        self.i
    }
}
"#,
        );
    }

    #[test]
    fn test_self_to_parameter() {
        check(
            "foo",
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(&mut $0self) -> i32 {
        self.i
    }
}
"#,
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(foo: &mut Foo) -> i32 {
        foo.i
    }
}
"#,
        );
    }

    #[test]
    fn test_owned_self_to_parameter() {
        cov_mark::check!(rename_self_to_param);
        check(
            "foo",
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f($0self) -> i32 {
        self.i
    }
}
"#,
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(foo: Foo) -> i32 {
        foo.i
    }
}
"#,
        );
    }

    #[test]
    fn test_self_in_path_to_parameter() {
        check(
            "foo",
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(&self) -> i32 {
        let self_var = 1;
        self$0.i
    }
}
"#,
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(foo: &Foo) -> i32 {
        let self_var = 1;
        foo.i
    }
}
"#,
        );
    }

    #[test]
    fn test_rename_field_put_init_shorthand() {
        cov_mark::check!(test_rename_field_put_init_shorthand);
        check(
            "bar",
            r#"
struct Foo { i$0: i32 }

fn foo(bar: i32) -> Foo {
    Foo { i: bar }
}
"#,
            r#"
struct Foo { bar: i32 }

fn foo(bar: i32) -> Foo {
    Foo { bar }
}
"#,
        );
    }

    #[test]
    fn test_rename_local_put_init_shorthand() {
        cov_mark::check!(test_rename_local_put_init_shorthand);
        check(
            "i",
            r#"
struct Foo { i: i32 }

fn foo(bar$0: i32) -> Foo {
    Foo { i: bar }
}
"#,
            r#"
struct Foo { i: i32 }

fn foo(i: i32) -> Foo {
    Foo { i }
}
"#,
        );
    }

    #[test]
    fn test_struct_field_pat_into_shorthand() {
        cov_mark::check!(test_rename_field_put_init_shorthand_pat);
        check(
            "baz",
            r#"
struct Foo { i$0: i32 }

fn foo(foo: Foo) {
    let Foo { i: ref baz @ qux } = foo;
    let _ = qux;
}
"#,
            r#"
struct Foo { baz: i32 }

fn foo(foo: Foo) {
    let Foo { ref baz @ qux } = foo;
    let _ = qux;
}
"#,
        );
    }

    #[test]
    fn test_rename_binding_in_destructure_pat() {
        let expected_fixture = r#"
struct Foo {
    i: i32,
}

fn foo(foo: Foo) {
    let Foo { i: bar } = foo;
    let _ = bar;
}
"#;
        check(
            "bar",
            r#"
struct Foo {
    i: i32,
}

fn foo(foo: Foo) {
    let Foo { i: b } = foo;
    let _ = b$0;
}
"#,
            expected_fixture,
        );
        check(
            "bar",
            r#"
struct Foo {
    i: i32,
}

fn foo(foo: Foo) {
    let Foo { i } = foo;
    let _ = i$0;
}
"#,
            expected_fixture,
        );
    }

    #[test]
    fn test_rename_binding_in_destructure_param_pat() {
        check(
            "bar",
            r#"
struct Foo {
    i: i32
}

fn foo(Foo { i }: foo) -> i32 {
    i$0
}
"#,
            r#"
struct Foo {
    i: i32
}

fn foo(Foo { i: bar }: foo) -> i32 {
    bar
}
"#,
        )
    }

    #[test]
    fn test_struct_field_complex_ident_pat() {
        check(
            "baz",
            r#"
struct Foo { i$0: i32 }

fn foo(foo: Foo) {
    let Foo { ref i } = foo;
}
"#,
            r#"
struct Foo { baz: i32 }

fn foo(foo: Foo) {
    let Foo { baz: ref i } = foo;
}
"#,
        );
    }

    #[test]
    fn test_rename_lifetimes() {
        cov_mark::check!(rename_lifetime);
        check(
            "'yeeee",
            r#"
trait Foo<'a> {
    fn foo() -> &'a ();
}
impl<'a> Foo<'a> for &'a () {
    fn foo() -> &'a$0 () {
        unimplemented!()
    }
}
"#,
            r#"
trait Foo<'a> {
    fn foo() -> &'a ();
}
impl<'yeeee> Foo<'yeeee> for &'yeeee () {
    fn foo() -> &'yeeee () {
        unimplemented!()
    }
}
"#,
        )
    }

    #[test]
    fn test_rename_bind_pat() {
        check(
            "new_name",
            r#"
fn main() {
    enum CustomOption<T> {
        None,
        Some(T),
    }

    let test_variable = CustomOption::Some(22);

    match test_variable {
        CustomOption::Some(foo$0) if foo == 11 => {}
        _ => (),
    }
}"#,
            r#"
fn main() {
    enum CustomOption<T> {
        None,
        Some(T),
    }

    let test_variable = CustomOption::Some(22);

    match test_variable {
        CustomOption::Some(new_name) if new_name == 11 => {}
        _ => (),
    }
}"#,
        );
    }

    #[test]
    fn test_rename_label() {
        check(
            "'foo",
            r#"
fn foo<'a>() -> &'a () {
    'a: {
        'b: loop {
            break 'a$0;
        }
    }
}
"#,
            r#"
fn foo<'a>() -> &'a () {
    'foo: {
        'b: loop {
            break 'foo;
        }
    }
}
"#,
        )
    }

    #[test]
    fn test_self_to_self() {
        cov_mark::check!(rename_self_to_self);
        check(
            "self",
            r#"
struct Foo;
impl Foo {
    fn foo(self$0) {}
}
"#,
            r#"
struct Foo;
impl Foo {
    fn foo(self) {}
}
"#,
        )
    }

    #[test]
    fn test_rename_field_in_pat_in_macro_doesnt_shorthand() {
        // ideally we would be able to make this emit a short hand, but I doubt this is easily possible
        check(
            "baz",
            r#"
macro_rules! foo {
    ($pattern:pat) => {
        let $pattern = loop {};
    };
}
struct Foo {
    bar$0: u32,
}
fn foo() {
    foo!(Foo { bar: baz });
}
"#,
            r#"
macro_rules! foo {
    ($pattern:pat) => {
        let $pattern = loop {};
    };
}
struct Foo {
    baz: u32,
}
fn foo() {
    foo!(Foo { baz: baz });
}
"#,
        )
    }

    #[test]
    fn test_rename_tuple_field() {
        check(
            "foo",
            r#"
struct Foo(i32);

fn baz() {
    let mut x = Foo(4);
    x.0$0 = 5;
}
"#,
            "error: No identifier available to rename",
        );
    }

    #[test]
    fn test_rename_builtin() {
        check(
            "foo",
            r#"
fn foo() {
    let x: i32$0 = 0;
}
"#,
            "error: Cannot rename builtin type",
        );
    }

    #[test]
    fn test_rename_self() {
        check(
            "foo",
            r#"
struct Foo {}

impl Foo {
    fn foo(self) -> Self$0 {
        self
    }
}
"#,
            "error: Cannot rename `Self`",
        );
    }
}
