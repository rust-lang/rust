//! Rename infrastructure for rust-analyzer. It is used primarily for the
//! literal "rename" in the ide (look for tests there), but it is also available
//! as a general-purpose service. For example, it is used by the fix for the
//! "incorrect case" diagnostic.
//!
//! It leverages the [`crate::search`] functionality to find what needs to be
//! renamed. The actual renames are tricky -- field shorthands need special
//! attention, and, when renaming modules, you also want to rename files on the
//! file system.
//!
//! Another can of worms are macros:
//!
//! ```
//! macro_rules! m { () => { fn f() {} } }
//! m!();
//! fn main() {
//!     f() // <- rename me
//! }
//! ```
//!
//! The correct behavior in such cases is probably to show a dialog to the user.
//! Our current behavior is ¯\_(ツ)_/¯.
use std::fmt;

use base_db::{AnchoredPathBuf, FileId, FileRange};
use either::Either;
use hir::{AsAssocItem, FieldSource, HasSource, InFile, ModuleSource, Semantics};
use stdx::never;
use syntax::{
    ast::{self, NameOwner},
    lex_single_syntax_kind, AstNode, SyntaxKind, TextRange, T,
};
use text_edit::TextEdit;

use crate::{
    defs::Definition,
    search::FileReference,
    source_change::{FileSystemEdit, SourceChange},
    RootDatabase,
};

pub type Result<T, E = RenameError> = std::result::Result<T, E>;

#[derive(Debug)]
pub struct RenameError(pub String);

impl fmt::Display for RenameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

#[macro_export]
macro_rules! _format_err {
    ($fmt:expr) => { RenameError(format!($fmt)) };
    ($fmt:expr, $($arg:tt)+) => { RenameError(format!($fmt, $($arg)+)) }
}
pub use _format_err as format_err;

#[macro_export]
macro_rules! _bail {
    ($($tokens:tt)*) => { return Err(format_err!($($tokens)*)) }
}
pub use _bail as bail;

impl Definition {
    pub fn rename(&self, sema: &Semantics<RootDatabase>, new_name: &str) -> Result<SourceChange> {
        match *self {
            Definition::ModuleDef(hir::ModuleDef::Module(module)) => {
                rename_mod(sema, module, new_name)
            }
            Definition::ModuleDef(hir::ModuleDef::BuiltinType(_)) => {
                bail!("Cannot rename builtin type")
            }
            Definition::SelfType(_) => bail!("Cannot rename `Self`"),
            def => rename_reference(sema, def, new_name),
        }
    }

    /// Textual range of the identifier which will change when renaming this
    /// `Definition`. Note that some definitions, like buitin types, can't be
    /// renamed.
    pub fn range_for_rename(self, sema: &Semantics<RootDatabase>) -> Option<FileRange> {
        // FIXME: the `original_file_range` calls here are wrong -- they never fail,
        // and _fall back_ to the entirety of the macro call. Such fall back is
        // incorrect for renames. The safe behavior would be to return an error for
        // such cases. The correct behavior would be to return an auxiliary list of
        // "can't rename these occurrences in macros" items, and then show some kind
        // of a dialog to the user. See:
        cov_mark::hit!(macros_are_broken_lol);

        let res = match self {
            Definition::Macro(mac) => {
                let src = mac.source(sema.db)?;
                let name = match &src.value {
                    Either::Left(it) => it.name()?,
                    Either::Right(it) => it.name()?,
                };
                src.with_value(name.syntax()).original_file_range(sema.db)
            }
            Definition::Field(field) => {
                let src = field.source(sema.db)?;

                match &src.value {
                    FieldSource::Named(record_field) => {
                        let name = record_field.name()?;
                        src.with_value(name.syntax()).original_file_range(sema.db)
                    }
                    FieldSource::Pos(_) => {
                        return None;
                    }
                }
            }
            Definition::ModuleDef(module_def) => match module_def {
                hir::ModuleDef::Module(module) => {
                    let src = module.declaration_source(sema.db)?;
                    let name = src.value.name()?;
                    src.with_value(name.syntax()).original_file_range(sema.db)
                }
                hir::ModuleDef::Function(it) => name_range(it, sema)?,
                hir::ModuleDef::Adt(adt) => match adt {
                    hir::Adt::Struct(it) => name_range(it, sema)?,
                    hir::Adt::Union(it) => name_range(it, sema)?,
                    hir::Adt::Enum(it) => name_range(it, sema)?,
                },
                hir::ModuleDef::Variant(it) => name_range(it, sema)?,
                hir::ModuleDef::Const(it) => name_range(it, sema)?,
                hir::ModuleDef::Static(it) => name_range(it, sema)?,
                hir::ModuleDef::Trait(it) => name_range(it, sema)?,
                hir::ModuleDef::TypeAlias(it) => name_range(it, sema)?,
                hir::ModuleDef::BuiltinType(_) => return None,
            },
            Definition::SelfType(_) => return None,
            Definition::Local(local) => {
                let src = local.source(sema.db);
                let name = match &src.value {
                    Either::Left(bind_pat) => bind_pat.name()?,
                    Either::Right(_) => return None,
                };
                src.with_value(name.syntax()).original_file_range(sema.db)
            }
            Definition::GenericParam(generic_param) => match generic_param {
                hir::GenericParam::TypeParam(type_param) => {
                    let src = type_param.source(sema.db)?;
                    let name = match &src.value {
                        Either::Left(type_param) => type_param.name()?,
                        Either::Right(_trait) => return None,
                    };
                    src.with_value(name.syntax()).original_file_range(sema.db)
                }
                hir::GenericParam::LifetimeParam(lifetime_param) => {
                    let src = lifetime_param.source(sema.db)?;
                    let lifetime = src.value.lifetime()?;
                    src.with_value(lifetime.syntax()).original_file_range(sema.db)
                }
                hir::GenericParam::ConstParam(it) => name_range(it, sema)?,
            },
            Definition::Label(label) => {
                let src = label.source(sema.db);
                let lifetime = src.value.lifetime()?;
                src.with_value(lifetime.syntax()).original_file_range(sema.db)
            }
        };
        return Some(res);

        fn name_range<D>(def: D, sema: &Semantics<RootDatabase>) -> Option<FileRange>
        where
            D: HasSource,
            D::Ast: ast::NameOwner,
        {
            let src = def.source(sema.db)?;
            let name = src.value.name()?;
            let res = src.with_value(name.syntax()).original_file_range(sema.db);
            Some(res)
        }
    }
}

fn rename_mod(
    sema: &Semantics<RootDatabase>,
    module: hir::Module,
    new_name: &str,
) -> Result<SourceChange> {
    if IdentifierKind::classify(new_name)? != IdentifierKind::Ident {
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
    let def = Definition::ModuleDef(hir::ModuleDef::Module(module));
    let usages = def.usages(sema).all();
    let ref_edits = usages.iter().map(|(&file_id, references)| {
        (file_id, source_edit_from_references(references, def, new_name))
    });
    source_change.extend(ref_edits);

    Ok(source_change)
}

fn rename_reference(
    sema: &Semantics<RootDatabase>,
    mut def: Definition,
    new_name: &str,
) -> Result<SourceChange> {
    let ident_kind = IdentifierKind::classify(new_name)?;

    if matches!(
        def, // is target a lifetime?
        Definition::GenericParam(hir::GenericParam::LifetimeParam(_)) | Definition::Label(_)
    ) {
        match ident_kind {
            IdentifierKind::Ident | IdentifierKind::Underscore => {
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
            (IdentifierKind::Ident, _) => cov_mark::hit!(rename_non_local),
            (IdentifierKind::Underscore, _) => (),
        }
    }

    def = match def {
        // HACK: resolve trait impl items to the item def of the trait definition
        // so that we properly resolve all trait item references
        Definition::ModuleDef(mod_def) => mod_def
            .as_assoc_item(sema.db)
            .and_then(|it| it.containing_trait_impl(sema.db))
            .and_then(|it| {
                it.items(sema.db).into_iter().find_map(|it| match (it, mod_def) {
                    (hir::AssocItem::Function(trait_func), hir::ModuleDef::Function(func))
                        if trait_func.name(sema.db) == func.name(sema.db) =>
                    {
                        Some(Definition::ModuleDef(hir::ModuleDef::Function(trait_func)))
                    }
                    (hir::AssocItem::Const(trait_konst), hir::ModuleDef::Const(konst))
                        if trait_konst.name(sema.db) == konst.name(sema.db) =>
                    {
                        Some(Definition::ModuleDef(hir::ModuleDef::Const(trait_konst)))
                    }
                    (
                        hir::AssocItem::TypeAlias(trait_type_alias),
                        hir::ModuleDef::TypeAlias(type_alias),
                    ) if trait_type_alias.name(sema.db) == type_alias.name(sema.db) => {
                        Some(Definition::ModuleDef(hir::ModuleDef::TypeAlias(trait_type_alias)))
                    }
                    _ => None,
                })
            })
            .unwrap_or(def),
        _ => def,
    };
    let usages = def.usages(sema).all();

    if !usages.is_empty() && ident_kind == IdentifierKind::Underscore {
        cov_mark::hit!(rename_underscore_multiple);
        bail!("Cannot rename reference to `_` as it is being referenced multiple times");
    }
    let mut source_change = SourceChange::default();
    source_change.extend(usages.iter().map(|(&file_id, references)| {
        (file_id, source_edit_from_references(references, def, new_name))
    }));

    let (file_id, edit) = source_edit_from_def(sema, def, new_name)?;
    source_change.insert_source_edit(file_id, edit);
    Ok(source_change)
}

pub fn source_edit_from_references(
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
) -> Result<(FileId, TextEdit)> {
    let frange = def
        .range_for_rename(sema)
        .ok_or_else(|| format_err!("No identifier available to rename"))?;

    let mut replacement_text = String::new();
    let mut repl_range = frange.range;
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
    Ok((frange.file_id, edit))
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum IdentifierKind {
    Ident,
    Lifetime,
    Underscore,
}

impl IdentifierKind {
    pub fn classify(new_name: &str) -> Result<IdentifierKind> {
        match lex_single_syntax_kind(new_name) {
            Some(res) => match res {
                (SyntaxKind::IDENT, _) => Ok(IdentifierKind::Ident),
                (T![_], _) => Ok(IdentifierKind::Underscore),
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
}
