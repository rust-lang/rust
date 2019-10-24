//! This module implements a reference search.
//! First, the element at the cursor position must be either an `ast::Name`
//! or `ast::NameRef`. If it's a `ast::NameRef`, at the classification step we
//! try to resolve the direct tree parent of this element, otherwise we
//! already have a definition and just need to get its HIR together with
//! some information that is needed for futher steps of searching.
//! After that, we collect files that might contain references and look
//! for text occurrences of the identifier. If there's an `ast::NameRef`
//! at the index that the match starts at and its tree parent is
//! resolved to the search element definition, we get a reference.

mod classify;
mod name_definition;
mod rename;
mod search_scope;

use once_cell::unsync::Lazy;
use ra_db::{SourceDatabase, SourceDatabaseExt};
use ra_prof::profile;
use ra_syntax::{algo::find_node_at_offset, ast, AstNode, SourceFile, SyntaxNode, TextUnit};

use crate::{db::RootDatabase, FilePosition, FileRange, NavigationTarget, RangeInfo};

pub(crate) use self::{
    classify::{classify_name, classify_name_ref},
    name_definition::{NameDefinition, NameKind},
    rename::rename,
};

pub use self::search_scope::SearchScope;

#[derive(Debug, Clone)]
pub struct ReferenceSearchResult {
    declaration: NavigationTarget,
    references: Vec<FileRange>,
}

impl ReferenceSearchResult {
    pub fn declaration(&self) -> &NavigationTarget {
        &self.declaration
    }

    pub fn references(&self) -> &[FileRange] {
        &self.references
    }

    /// Total number of references
    /// At least 1 since all valid references should
    /// Have a declaration
    pub fn len(&self) -> usize {
        self.references.len() + 1
    }
}

// allow turning ReferenceSearchResult into an iterator
// over FileRanges
impl IntoIterator for ReferenceSearchResult {
    type Item = FileRange;
    type IntoIter = std::vec::IntoIter<FileRange>;

    fn into_iter(mut self) -> Self::IntoIter {
        let mut v = Vec::with_capacity(self.len());
        v.push(FileRange { file_id: self.declaration.file_id(), range: self.declaration.range() });
        v.append(&mut self.references);
        v.into_iter()
    }
}

pub(crate) fn find_all_refs(
    db: &RootDatabase,
    position: FilePosition,
    search_scope: Option<SearchScope>,
) -> Option<RangeInfo<ReferenceSearchResult>> {
    let parse = db.parse(position.file_id);
    let syntax = parse.tree().syntax().clone();
    let RangeInfo { range, info: (name, def) } = find_name(db, &syntax, position)?;

    let declaration = match def.kind {
        NameKind::Macro(mac) => NavigationTarget::from_macro_def(db, mac),
        NameKind::Field(field) => NavigationTarget::from_field(db, field),
        NameKind::AssocItem(assoc) => NavigationTarget::from_assoc_item(db, assoc),
        NameKind::Def(def) => NavigationTarget::from_def(db, def)?,
        NameKind::SelfType(ref ty) => match ty.as_adt() {
            Some((def_id, _)) => NavigationTarget::from_adt_def(db, def_id),
            None => return None,
        },
        NameKind::Pat((_, pat)) => NavigationTarget::from_pat(db, position.file_id, pat),
        NameKind::SelfParam(par) => NavigationTarget::from_self_param(position.file_id, par),
        NameKind::GenericParam(_) => return None,
    };

    let search_scope = {
        let base = def.search_scope(db);
        match search_scope {
            None => base,
            Some(scope) => base.intersection(&scope),
        }
    };

    let references = process_definition(db, def, name, search_scope);

    Some(RangeInfo::new(range, ReferenceSearchResult { declaration, references }))
}

fn find_name<'a>(
    db: &RootDatabase,
    syntax: &SyntaxNode,
    position: FilePosition,
) -> Option<RangeInfo<(String, NameDefinition)>> {
    if let Some(name) = find_node_at_offset::<ast::Name>(&syntax, position.offset) {
        let def = classify_name(db, position.file_id, &name)?;
        let range = name.syntax().text_range();
        return Some(RangeInfo::new(range, (name.text().to_string(), def)));
    }
    let name_ref = find_node_at_offset::<ast::NameRef>(&syntax, position.offset)?;
    let def = classify_name_ref(db, position.file_id, &name_ref)?;
    let range = name_ref.syntax().text_range();
    Some(RangeInfo::new(range, (name_ref.text().to_string(), def)))
}

fn process_definition(
    db: &RootDatabase,
    def: NameDefinition,
    name: String,
    scope: SearchScope,
) -> Vec<FileRange> {
    let _p = profile("process_definition");

    let pat = name.as_str();
    let mut refs = vec![];

    for (file_id, search_range) in scope {
        let text = db.file_text(file_id);
        let parse = Lazy::new(|| SourceFile::parse(&text));

        for (idx, _) in text.match_indices(pat) {
            let offset = TextUnit::from_usize(idx);

            if let Some(name_ref) =
                find_node_at_offset::<ast::NameRef>(parse.tree().syntax(), offset)
            {
                let range = name_ref.syntax().text_range();
                if let Some(search_range) = search_range {
                    if !range.is_subrange(&search_range) {
                        continue;
                    }
                }
                if let Some(d) = classify_name_ref(db, file_id, &name_ref) {
                    if d == def {
                        refs.push(FileRange { file_id, range });
                    }
                }
            }
        }
    }
    refs
}

#[cfg(test)]
mod tests {
    use crate::{
        mock_analysis::{analysis_and_position, single_file_with_position, MockAnalysis},
        ReferenceSearchResult, SearchScope,
    };

    #[test]
    fn test_find_all_refs_for_local() {
        let code = r#"
    fn main() {
        let mut i = 1;
        let j = 1;
        i = i<|> + j;

        {
            i = 0;
        }

        i = 5;
    }"#;

        let refs = get_all_refs(code);
        assert_eq!(refs.len(), 5);
    }

    #[test]
    fn test_find_all_refs_for_param_inside() {
        let code = r#"
    fn foo(i : u32) -> u32 {
        i<|>
    }"#;

        let refs = get_all_refs(code);
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn test_find_all_refs_for_fn_param() {
        let code = r#"
    fn foo(i<|> : u32) -> u32 {
        i
    }"#;

        let refs = get_all_refs(code);
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn test_find_all_refs_field_name() {
        let code = r#"
            //- /lib.rs
            struct Foo {
                pub spam<|>: u32,
            }

            fn main(s: Foo) {
                let f = s.spam;
            }
        "#;

        let refs = get_all_refs(code);
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn test_find_all_refs_impl_item_name() {
        let code = r#"
            //- /lib.rs
            struct Foo;
            impl Foo {
                fn f<|>(&self) {  }
            }
        "#;

        let refs = get_all_refs(code);
        assert_eq!(refs.len(), 1);
    }

    #[test]
    fn test_find_all_refs_enum_var_name() {
        let code = r#"
            //- /lib.rs
            enum Foo {
                A,
                B<|>,
                C,
            }
        "#;

        let refs = get_all_refs(code);
        assert_eq!(refs.len(), 1);
    }

    #[test]
    fn test_find_all_refs_two_modules() {
        let code = r#"
            //- /lib.rs
            pub mod foo;
            pub mod bar;

            fn f() {
                let i = foo::Foo { n: 5 };
            }

            //- /foo.rs
            use crate::bar;

            pub struct Foo {
                pub n: u32,
            }

            fn f() {
                let i = bar::Bar { n: 5 };
            }

            //- /bar.rs
            use crate::foo;

            pub struct Bar {
                pub n: u32,
            }

            fn f() {
                let i = foo::Foo<|> { n: 5 };
            }
        "#;

        let (analysis, pos) = analysis_and_position(code);
        let refs = analysis.find_all_refs(pos, None).unwrap().unwrap();
        assert_eq!(refs.len(), 3);
    }

    // `mod foo;` is not in the results because `foo` is an `ast::Name`.
    // So, there are two references: the first one is a definition of the `foo` module,
    // which is the whole `foo.rs`, and the second one is in `use foo::Foo`.
    #[test]
    fn test_find_all_refs_decl_module() {
        let code = r#"
            //- /lib.rs
            mod foo<|>;

            use foo::Foo;

            fn f() {
                let i = Foo { n: 5 };
            }

            //- /foo.rs
            pub struct Foo {
                pub n: u32,
            }
        "#;

        let (analysis, pos) = analysis_and_position(code);
        let refs = analysis.find_all_refs(pos, None).unwrap().unwrap();
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn test_find_all_refs_super_mod_vis() {
        let code = r#"
            //- /lib.rs
            mod foo;

            //- /foo.rs
            mod some;
            use some::Foo;

            fn f() {
                let i = Foo { n: 5 };
            }

            //- /foo/some.rs
            pub(super) struct Foo<|> {
                pub n: u32,
            }
        "#;

        let (analysis, pos) = analysis_and_position(code);
        let refs = analysis.find_all_refs(pos, None).unwrap().unwrap();
        assert_eq!(refs.len(), 3);
    }

    #[test]
    fn test_find_all_refs_with_scope() {
        let code = r#"
            //- /lib.rs
            mod foo;
            mod bar;

            pub fn quux<|>() {}

            //- /foo.rs
            fn f() { super::quux(); }

            //- /bar.rs
            fn f() { super::quux(); }
        "#;

        let (mock, pos) = MockAnalysis::with_files_and_position(code);
        let bar = mock.id_of("/bar.rs");
        let analysis = mock.analysis();

        let refs = analysis.find_all_refs(pos, None).unwrap().unwrap();
        assert_eq!(refs.len(), 3);

        let refs =
            analysis.find_all_refs(pos, Some(SearchScope::single_file(bar))).unwrap().unwrap();
        assert_eq!(refs.len(), 2);
    }

    fn get_all_refs(text: &str) -> ReferenceSearchResult {
        let (analysis, position) = single_file_with_position(text);
        analysis.find_all_refs(position, None).unwrap().unwrap()
    }
}
