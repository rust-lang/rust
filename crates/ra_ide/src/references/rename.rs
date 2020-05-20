//! FIXME: write short doc here

use hir::{ModuleSource, Semantics};
use ra_db::{RelativePath, RelativePathBuf, SourceDatabaseExt};
use ra_ide_db::RootDatabase;
use ra_syntax::{
    algo::find_node_at_offset, ast, ast::TypeAscriptionOwner, lex_single_valid_syntax_kind,
    AstNode, SyntaxKind, SyntaxNode, SyntaxToken,
};
use ra_text_edit::TextEdit;
use std::convert::TryInto;
use test_utils::mark;

use crate::{
    references::find_all_refs, FilePosition, FileSystemEdit, RangeInfo, Reference, ReferenceKind,
    SourceChange, SourceFileEdit, TextRange, TextSize,
};

pub(crate) fn rename(
    db: &RootDatabase,
    position: FilePosition,
    new_name: &str,
) -> Option<RangeInfo<SourceChange>> {
    match lex_single_valid_syntax_kind(new_name)? {
        SyntaxKind::IDENT | SyntaxKind::UNDERSCORE => (),
        SyntaxKind::SELF_KW => return rename_to_self(db, position),
        _ => return None,
    }

    let sema = Semantics::new(db);
    let source_file = sema.parse(position.file_id);
    let syntax = source_file.syntax();
    if let Some((ast_name, ast_module)) = find_name_and_module_at_offset(syntax, position) {
        let range = ast_name.syntax().text_range();
        rename_mod(&sema, &ast_name, &ast_module, position, new_name)
            .map(|info| RangeInfo::new(range, info))
    } else if let Some(self_token) =
        syntax.token_at_offset(position.offset).find(|t| t.kind() == SyntaxKind::SELF_KW)
    {
        rename_self_to_param(db, position, self_token, new_name)
    } else {
        rename_reference(sema.db, position, new_name)
    }
}

fn find_name_and_module_at_offset(
    syntax: &SyntaxNode,
    position: FilePosition,
) -> Option<(ast::Name, ast::Module)> {
    let ast_name = find_node_at_offset::<ast::Name>(syntax, position.offset)?;
    let ast_module = ast::Module::cast(ast_name.syntax().parent()?)?;
    Some((ast_name, ast_module))
}

fn source_edit_from_reference(reference: Reference, new_name: &str) -> SourceFileEdit {
    let mut replacement_text = String::new();
    let file_id = reference.file_range.file_id;
    let range = match reference.kind {
        ReferenceKind::FieldShorthandForField => {
            mark::hit!(test_rename_struct_field_for_shorthand);
            replacement_text.push_str(new_name);
            replacement_text.push_str(": ");
            TextRange::new(reference.file_range.range.start(), reference.file_range.range.start())
        }
        ReferenceKind::FieldShorthandForLocal => {
            mark::hit!(test_rename_local_for_field_shorthand);
            replacement_text.push_str(": ");
            replacement_text.push_str(new_name);
            TextRange::new(reference.file_range.range.end(), reference.file_range.range.end())
        }
        _ => {
            replacement_text.push_str(new_name);
            reference.file_range.range
        }
    };
    SourceFileEdit { file_id, edit: TextEdit::replace(range, replacement_text) }
}

fn rename_mod(
    sema: &Semantics<RootDatabase>,
    ast_name: &ast::Name,
    ast_module: &ast::Module,
    position: FilePosition,
    new_name: &str,
) -> Option<SourceChange> {
    let mut source_file_edits = Vec::new();
    let mut file_system_edits = Vec::new();
    if let Some(module) = sema.to_def(ast_module) {
        let src = module.definition_source(sema.db);
        let file_id = src.file_id.original_file(sema.db);
        match src.value {
            ModuleSource::SourceFile(..) => {
                let mod_path: RelativePathBuf = sema.db.file_relative_path(file_id);
                // mod is defined in path/to/dir/mod.rs
                let dst_path = if mod_path.file_stem() == Some("mod") {
                    mod_path
                        .parent()
                        .and_then(|p| p.parent())
                        .or_else(|| Some(RelativePath::new("")))
                        .map(|p| p.join(new_name).join("mod.rs"))
                } else {
                    Some(mod_path.with_file_name(new_name).with_extension("rs"))
                };
                if let Some(path) = dst_path {
                    let move_file = FileSystemEdit::MoveFile {
                        src: file_id,
                        dst_source_root: sema.db.file_source_root(position.file_id),
                        dst_path: path,
                    };
                    file_system_edits.push(move_file);
                }
            }
            ModuleSource::Module(..) => {}
        }
    }

    let edit = SourceFileEdit {
        file_id: position.file_id,
        edit: TextEdit::replace(ast_name.syntax().text_range(), new_name.into()),
    };
    source_file_edits.push(edit);

    if let Some(RangeInfo { range: _, info: refs }) = find_all_refs(sema.db, position, None) {
        let ref_edits = refs
            .references
            .into_iter()
            .map(|reference| source_edit_from_reference(reference, new_name));
        source_file_edits.extend(ref_edits);
    }

    Some(SourceChange::from_edits("Rename", source_file_edits, file_system_edits))
}

fn rename_to_self(db: &RootDatabase, position: FilePosition) -> Option<RangeInfo<SourceChange>> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(position.file_id);
    let syn = source_file.syntax();

    let fn_def = find_node_at_offset::<ast::FnDef>(syn, position.offset)?;
    let params = fn_def.param_list()?;
    if params.self_param().is_some() {
        return None; // method already has self param
    }
    let first_param = params.params().next()?;
    let mutable = match first_param.ascribed_type() {
        Some(ast::TypeRef::ReferenceType(rt)) => rt.mut_token().is_some(),
        _ => return None, // not renaming other types
    };

    let RangeInfo { range, info: refs } = find_all_refs(db, position, None)?;

    let param_range = first_param.syntax().text_range();
    let (param_ref, usages): (Vec<Reference>, Vec<Reference>) = refs
        .into_iter()
        .partition(|reference| param_range.intersect(reference.file_range.range).is_some());

    if param_ref.is_empty() {
        return None;
    }

    let mut edits = usages
        .into_iter()
        .map(|reference| source_edit_from_reference(reference, "self"))
        .collect::<Vec<_>>();

    edits.push(SourceFileEdit {
        file_id: position.file_id,
        edit: TextEdit::replace(
            param_range,
            String::from(if mutable { "&mut self" } else { "&self" }),
        ),
    });

    Some(RangeInfo::new(range, SourceChange::source_file_edits("Rename", edits)))
}

fn text_edit_from_self_param(
    syn: &SyntaxNode,
    self_param: &ast::SelfParam,
    new_name: &str,
) -> Option<TextEdit> {
    fn target_type_name(impl_def: &ast::ImplDef) -> Option<String> {
        if let Some(ast::TypeRef::PathType(p)) = impl_def.target_type() {
            return Some(p.path()?.segment()?.name_ref()?.text().to_string());
        }
        None
    }

    let impl_def =
        find_node_at_offset::<ast::ImplDef>(syn, self_param.syntax().text_range().start())?;
    let type_name = target_type_name(&impl_def)?;

    let mut replacement_text = String::from(new_name);
    replacement_text.push_str(": ");
    replacement_text.push_str(self_param.mut_token().map_or("&", |_| "&mut "));
    replacement_text.push_str(type_name.as_str());

    Some(TextEdit::replace(self_param.syntax().text_range(), replacement_text))
}

fn rename_self_to_param(
    db: &RootDatabase,
    position: FilePosition,
    self_token: SyntaxToken,
    new_name: &str,
) -> Option<RangeInfo<SourceChange>> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(position.file_id);
    let syn = source_file.syntax();

    let text = db.file_text(position.file_id);
    let fn_def = find_node_at_offset::<ast::FnDef>(syn, position.offset)?;
    let search_range = fn_def.syntax().text_range();

    let mut edits: Vec<SourceFileEdit> = vec![];

    for (idx, _) in text.match_indices("self") {
        let offset: TextSize = idx.try_into().unwrap();
        if !search_range.contains_inclusive(offset) {
            continue;
        }
        if let Some(ref usage) =
            syn.token_at_offset(offset).find(|t| t.kind() == SyntaxKind::SELF_KW)
        {
            let edit = if let Some(ref self_param) = ast::SelfParam::cast(usage.parent()) {
                text_edit_from_self_param(syn, self_param, new_name)?
            } else {
                TextEdit::replace(usage.text_range(), String::from(new_name))
            };
            edits.push(SourceFileEdit { file_id: position.file_id, edit });
        }
    }

    let range = ast::SelfParam::cast(self_token.parent())
        .map_or(self_token.text_range(), |p| p.syntax().text_range());

    Some(RangeInfo::new(range, SourceChange::source_file_edits("Rename", edits)))
}

fn rename_reference(
    db: &RootDatabase,
    position: FilePosition,
    new_name: &str,
) -> Option<RangeInfo<SourceChange>> {
    let RangeInfo { range, info: refs } = find_all_refs(db, position, None)?;

    let edit = refs
        .into_iter()
        .map(|reference| source_edit_from_reference(reference, new_name))
        .collect::<Vec<_>>();

    if edit.is_empty() {
        return None;
    }

    Some(RangeInfo::new(range, SourceChange::source_file_edits("Rename", edit)))
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;
    use ra_text_edit::TextEditBuilder;
    use test_utils::{assert_eq_text, mark};

    use crate::{
        mock_analysis::analysis_and_position, mock_analysis::single_file_with_position, FileId,
    };

    #[test]
    fn test_rename_to_underscore() {
        test_rename(
            r#"
    fn main() {
        let i<|> = 1;
    }"#,
            "_",
            r#"
    fn main() {
        let _ = 1;
    }"#,
        );
    }

    #[test]
    fn test_rename_to_raw_identifier() {
        test_rename(
            r#"
    fn main() {
        let i<|> = 1;
    }"#,
            "r#fn",
            r#"
    fn main() {
        let r#fn = 1;
    }"#,
        );
    }

    #[test]
    fn test_rename_to_invalid_identifier() {
        let (analysis, position) = single_file_with_position(
            "
    fn main() {
        let i<|> = 1;
    }",
        );
        let new_name = "invalid!";
        let source_change = analysis.rename(position, new_name).unwrap();
        assert!(source_change.is_none());
    }

    #[test]
    fn test_rename_for_local() {
        test_rename(
            r#"
    fn main() {
        let mut i = 1;
        let j = 1;
        i = i<|> + j;

        {
            i = 0;
        }

        i = 5;
    }"#,
            "k",
            r#"
    fn main() {
        let mut k = 1;
        let j = 1;
        k = k + j;

        {
            k = 0;
        }

        k = 5;
    }"#,
        );
    }

    #[test]
    fn test_rename_for_macro_args() {
        test_rename(
            r#"
    macro_rules! foo {($i:ident) => {$i} }
    fn main() {
        let a<|> = "test";
        foo!(a);
    }"#,
            "b",
            r#"
    macro_rules! foo {($i:ident) => {$i} }
    fn main() {
        let b = "test";
        foo!(b);
    }"#,
        );
    }

    #[test]
    fn test_rename_for_macro_args_rev() {
        test_rename(
            r#"
    macro_rules! foo {($i:ident) => {$i} }
    fn main() {
        let a = "test";
        foo!(a<|>);
    }"#,
            "b",
            r#"
    macro_rules! foo {($i:ident) => {$i} }
    fn main() {
        let b = "test";
        foo!(b);
    }"#,
        );
    }

    #[test]
    fn test_rename_for_macro_define_fn() {
        test_rename(
            r#"
    macro_rules! define_fn {($id:ident) => { fn $id{} }}
    define_fn!(foo);
    fn main() {
        fo<|>o();
    }"#,
            "bar",
            r#"
    macro_rules! define_fn {($id:ident) => { fn $id{} }}
    define_fn!(bar);
    fn main() {
        bar();
    }"#,
        );
    }

    #[test]
    fn test_rename_for_macro_define_fn_rev() {
        test_rename(
            r#"
    macro_rules! define_fn {($id:ident) => { fn $id{} }}
    define_fn!(fo<|>o);
    fn main() {
        foo();
    }"#,
            "bar",
            r#"
    macro_rules! define_fn {($id:ident) => { fn $id{} }}
    define_fn!(bar);
    fn main() {
        bar();
    }"#,
        );
    }

    #[test]
    fn test_rename_for_param_inside() {
        test_rename(
            r#"
    fn foo(i : u32) -> u32 {
        i<|>
    }"#,
            "j",
            r#"
    fn foo(j : u32) -> u32 {
        j
    }"#,
        );
    }

    #[test]
    fn test_rename_refs_for_fn_param() {
        test_rename(
            r#"
    fn foo(i<|> : u32) -> u32 {
        i
    }"#,
            "new_name",
            r#"
    fn foo(new_name : u32) -> u32 {
        new_name
    }"#,
        );
    }

    #[test]
    fn test_rename_for_mut_param() {
        test_rename(
            r#"
    fn foo(mut i<|> : u32) -> u32 {
        i
    }"#,
            "new_name",
            r#"
    fn foo(mut new_name : u32) -> u32 {
        new_name
    }"#,
        );
    }

    #[test]
    fn test_rename_struct_field() {
        test_rename(
            r#"
    struct Foo {
        i<|>: i32,
    }

    impl Foo {
        fn new(i: i32) -> Self {
            Self { i: i }
        }
    }
    "#,
            "j",
            r#"
    struct Foo {
        j: i32,
    }

    impl Foo {
        fn new(i: i32) -> Self {
            Self { j: i }
        }
    }
    "#,
        );
    }

    #[test]
    fn test_rename_struct_field_for_shorthand() {
        mark::check!(test_rename_struct_field_for_shorthand);
        test_rename(
            r#"
    struct Foo {
        i<|>: i32,
    }

    impl Foo {
        fn new(i: i32) -> Self {
            Self { i }
        }
    }
    "#,
            "j",
            r#"
    struct Foo {
        j: i32,
    }

    impl Foo {
        fn new(i: i32) -> Self {
            Self { j: i }
        }
    }
    "#,
        );
    }

    #[test]
    fn test_rename_local_for_field_shorthand() {
        mark::check!(test_rename_local_for_field_shorthand);
        test_rename(
            r#"
    struct Foo {
        i: i32,
    }

    impl Foo {
        fn new(i<|>: i32) -> Self {
            Self { i }
        }
    }
    "#,
            "j",
            r#"
    struct Foo {
        i: i32,
    }

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
        test_rename(
            r#"
    struct Foo {
        i<|>: i32,
    }

    struct Bar {
        i: i32,
    }

    impl Bar {
        fn new(i: i32) -> Self {
            Self { i }
        }
    }
    "#,
            "j",
            r#"
    struct Foo {
        j: i32,
    }

    struct Bar {
        i: i32,
    }

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
        test_rename(
            r#"
    struct Foo {
        i: i32,
    }

    fn baz(i<|>: i32) -> Self {
         let x = Foo { i };
         {
             let i = 0;
             Foo { i }
         }
     }
    "#,
            "j",
            r#"
    struct Foo {
        i: i32,
    }

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
        let (analysis, position) = analysis_and_position(
            "
            //- /lib.rs
            mod bar;

            //- /bar.rs
            mod foo<|>;

            //- /bar/foo.rs
            // emtpy
            ",
        );
        let new_name = "foo2";
        let source_change = analysis.rename(position, new_name).unwrap();
        assert_debug_snapshot!(&source_change,
@r###"
        Some(
            RangeInfo {
                range: 4..7,
                info: SourceChange {
                    label: "Rename",
                    source_file_edits: [
                        SourceFileEdit {
                            file_id: FileId(
                                2,
                            ),
                            edit: TextEdit {
                                indels: [
                                    Indel {
                                        insert: "foo2",
                                        delete: 4..7,
                                    },
                                ],
                            },
                        },
                    ],
                    file_system_edits: [
                        MoveFile {
                            src: FileId(
                                3,
                            ),
                            dst_source_root: SourceRootId(
                                0,
                            ),
                            dst_path: "bar/foo2.rs",
                        },
                    ],
                    cursor_position: None,
                    is_snippet: false,
                },
            },
        )
        "###);
    }

    #[test]
    fn test_rename_mod_in_dir() {
        let (analysis, position) = analysis_and_position(
            "
            //- /lib.rs
            mod fo<|>o;
            //- /foo/mod.rs
            // emtpy
            ",
        );
        let new_name = "foo2";
        let source_change = analysis.rename(position, new_name).unwrap();
        assert_debug_snapshot!(&source_change,
        @r###"
        Some(
            RangeInfo {
                range: 4..7,
                info: SourceChange {
                    label: "Rename",
                    source_file_edits: [
                        SourceFileEdit {
                            file_id: FileId(
                                1,
                            ),
                            edit: TextEdit {
                                indels: [
                                    Indel {
                                        insert: "foo2",
                                        delete: 4..7,
                                    },
                                ],
                            },
                        },
                    ],
                    file_system_edits: [
                        MoveFile {
                            src: FileId(
                                2,
                            ),
                            dst_source_root: SourceRootId(
                                0,
                            ),
                            dst_path: "foo2/mod.rs",
                        },
                    ],
                    cursor_position: None,
                    is_snippet: false,
                },
            },
        )
        "###
               );
    }

    #[test]
    fn test_module_rename_in_path() {
        test_rename(
            r#"
    mod <|>foo {
        pub fn bar() {}
    }

    fn main() {
        foo::bar();
    }"#,
            "baz",
            r#"
    mod baz {
        pub fn bar() {}
    }

    fn main() {
        baz::bar();
    }"#,
        );
    }

    #[test]
    fn test_rename_mod_filename_and_path() {
        let (analysis, position) = analysis_and_position(
            "
            //- /lib.rs
            mod bar;
            fn f() {
                bar::foo::fun()
            }

            //- /bar.rs
            pub mod foo<|>;

            //- /bar/foo.rs
            // pub fn fun() {}
            ",
        );
        let new_name = "foo2";
        let source_change = analysis.rename(position, new_name).unwrap();
        assert_debug_snapshot!(&source_change,
@r###"
        Some(
            RangeInfo {
                range: 8..11,
                info: SourceChange {
                    label: "Rename",
                    source_file_edits: [
                        SourceFileEdit {
                            file_id: FileId(
                                2,
                            ),
                            edit: TextEdit {
                                indels: [
                                    Indel {
                                        insert: "foo2",
                                        delete: 8..11,
                                    },
                                ],
                            },
                        },
                        SourceFileEdit {
                            file_id: FileId(
                                1,
                            ),
                            edit: TextEdit {
                                indels: [
                                    Indel {
                                        insert: "foo2",
                                        delete: 27..30,
                                    },
                                ],
                            },
                        },
                    ],
                    file_system_edits: [
                        MoveFile {
                            src: FileId(
                                3,
                            ),
                            dst_source_root: SourceRootId(
                                0,
                            ),
                            dst_path: "bar/foo2.rs",
                        },
                    ],
                    cursor_position: None,
                    is_snippet: false,
                },
            },
        )
        "###);
    }

    #[test]
    fn test_enum_variant_from_module_1() {
        test_rename(
            r#"
    mod foo {
        pub enum Foo {
            Bar<|>,
        }
    }

    fn func(f: foo::Foo) {
        match f {
            foo::Foo::Bar => {}
        }
    }
    "#,
            "Baz",
            r#"
    mod foo {
        pub enum Foo {
            Baz,
        }
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
        test_rename(
            r#"
    mod foo {
        pub struct Foo {
            pub bar<|>: uint,
        }
    }

    fn foo(f: foo::Foo) {
        let _ = f.bar;
    }
    "#,
            "baz",
            r#"
    mod foo {
        pub struct Foo {
            pub baz: uint,
        }
    }

    fn foo(f: foo::Foo) {
        let _ = f.baz;
    }
    "#,
        );
    }

    #[test]
    fn test_parameter_to_self() {
        test_rename(
            r#"
    struct Foo {
        i: i32,
    }

    impl Foo {
        fn f(foo<|>: &mut Foo) -> i32 {
            foo.i
        }
    }
    "#,
            "self",
            r#"
    struct Foo {
        i: i32,
    }

    impl Foo {
        fn f(&mut self) -> i32 {
            self.i
        }
    }
    "#,
        );
    }

    #[test]
    fn test_self_to_parameter() {
        test_rename(
            r#"
    struct Foo {
        i: i32,
    }

    impl Foo {
        fn f(&mut <|>self) -> i32 {
            self.i
        }
    }
    "#,
            "foo",
            r#"
    struct Foo {
        i: i32,
    }

    impl Foo {
        fn f(foo: &mut Foo) -> i32 {
            foo.i
        }
    }
    "#,
        );
    }

    #[test]
    fn test_self_in_path_to_parameter() {
        test_rename(
            r#"
    struct Foo {
        i: i32,
    }

    impl Foo {
        fn f(&self) -> i32 {
            let self_var = 1;
            self<|>.i
        }
    }
    "#,
            "foo",
            r#"
    struct Foo {
        i: i32,
    }

    impl Foo {
        fn f(foo: &Foo) -> i32 {
            let self_var = 1;
            foo.i
        }
    }
    "#,
        );
    }

    fn test_rename(text: &str, new_name: &str, expected: &str) {
        let (analysis, position) = single_file_with_position(text);
        let source_change = analysis.rename(position, new_name).unwrap();
        let mut text_edit_builder = TextEditBuilder::default();
        let mut file_id: Option<FileId> = None;
        if let Some(change) = source_change {
            for edit in change.info.source_file_edits {
                file_id = Some(edit.file_id);
                for indel in edit.edit.as_indels() {
                    text_edit_builder.replace(indel.delete, indel.insert.clone());
                }
            }
        }
        let mut result = analysis.file_text(file_id.unwrap()).unwrap().to_string();
        text_edit_builder.finish().apply(&mut result);
        assert_eq_text!(expected, &*result);
    }
}
