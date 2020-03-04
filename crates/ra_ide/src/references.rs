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

mod rename;
mod search_scope;

use hir::Semantics;
use once_cell::unsync::Lazy;
use ra_db::SourceDatabaseExt;
use ra_ide_db::{
    defs::{classify_name, classify_name_ref, Definition},
    RootDatabase,
};
use ra_prof::profile;
use ra_syntax::{
    algo::find_node_at_offset,
    ast::{self, NameOwner},
    match_ast, AstNode, SyntaxKind, SyntaxNode, TextRange, TextUnit, TokenAtOffset,
};
use test_utils::tested_by;

use crate::{display::TryToNav, FilePosition, FileRange, NavigationTarget, RangeInfo};

pub(crate) use self::rename::rename;

pub use ra_ide_db::search::SearchScope;

#[derive(Debug, Clone)]
pub struct ReferenceSearchResult {
    declaration: Declaration,
    references: Vec<Reference>,
}

#[derive(Debug, Clone)]
pub struct Declaration {
    pub nav: NavigationTarget,
    pub kind: ReferenceKind,
    pub access: Option<ReferenceAccess>,
}

#[derive(Debug, Clone)]
pub struct Reference {
    pub file_range: FileRange,
    pub kind: ReferenceKind,
    pub access: Option<ReferenceAccess>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReferenceKind {
    StructLiteral,
    Other,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ReferenceAccess {
    Read,
    Write,
}

impl ReferenceSearchResult {
    pub fn declaration(&self) -> &Declaration {
        &self.declaration
    }

    pub fn decl_target(&self) -> &NavigationTarget {
        &self.declaration.nav
    }

    pub fn references(&self) -> &[Reference] {
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
// over References
impl IntoIterator for ReferenceSearchResult {
    type Item = Reference;
    type IntoIter = std::vec::IntoIter<Reference>;

    fn into_iter(mut self) -> Self::IntoIter {
        let mut v = Vec::with_capacity(self.len());
        v.push(Reference {
            file_range: FileRange {
                file_id: self.declaration.nav.file_id(),
                range: self.declaration.nav.range(),
            },
            kind: self.declaration.kind,
            access: self.declaration.access,
        });
        v.append(&mut self.references);
        v.into_iter()
    }
}

pub(crate) fn find_all_refs(
    db: &RootDatabase,
    position: FilePosition,
    search_scope: Option<SearchScope>,
) -> Option<RangeInfo<ReferenceSearchResult>> {
    let _p = profile("find_all_refs");
    let sema = Semantics::new(db);
    let syntax = sema.parse(position.file_id).syntax().clone();

    let (opt_name, search_kind) =
        if let Some(name) = get_struct_def_name_for_struc_litetal_search(&syntax, position) {
            (Some(name), ReferenceKind::StructLiteral)
        } else {
            (find_node_at_offset::<ast::Name>(&syntax, position.offset), ReferenceKind::Other)
        };

    let RangeInfo { range, info: def } = find_name(&sema, &syntax, position, opt_name)?;

    let references = find_refs_to_def(db, &def, search_scope)
        .into_iter()
        .filter(|r| search_kind == ReferenceKind::Other || search_kind == r.kind)
        .collect();

    let decl_range = def.try_to_nav(db)?.range();

    let declaration = Declaration {
        nav: def.try_to_nav(db)?,
        kind: ReferenceKind::Other,
        access: decl_access(&def, &syntax, decl_range),
    };

    Some(RangeInfo::new(range, ReferenceSearchResult { declaration, references }))
}

pub(crate) fn find_refs_to_def(
    db: &RootDatabase,
    def: &Definition,
    search_scope: Option<SearchScope>,
) -> Vec<Reference> {
    let search_scope = {
        let base = SearchScope::for_def(&def, db);
        match search_scope {
            None => base,
            Some(scope) => base.intersection(&scope),
        }
    };

    let name = match def.name(db) {
        None => return Vec::new(),
        Some(it) => it.to_string(),
    };

    process_definition(db, def, name, search_scope)
}

fn find_name(
    sema: &Semantics<RootDatabase>,
    syntax: &SyntaxNode,
    position: FilePosition,
    opt_name: Option<ast::Name>,
) -> Option<RangeInfo<Definition>> {
    if let Some(name) = opt_name {
        let def = classify_name(sema, &name)?.definition();
        let range = name.syntax().text_range();
        return Some(RangeInfo::new(range, def));
    }
    let name_ref = find_node_at_offset::<ast::NameRef>(&syntax, position.offset)?;
    let def = classify_name_ref(sema, &name_ref)?.definition();
    let range = name_ref.syntax().text_range();
    Some(RangeInfo::new(range, def))
}

fn process_definition(
    db: &RootDatabase,
    def: &Definition,
    name: String,
    scope: SearchScope,
) -> Vec<Reference> {
    let _p = profile("process_definition");

    let pat = name.as_str();
    let mut refs = vec![];

    for (file_id, search_range) in scope {
        let text = db.file_text(file_id);
        let search_range =
            search_range.unwrap_or(TextRange::offset_len(0.into(), TextUnit::of_str(&text)));

        let sema = Semantics::new(db);
        let tree = Lazy::new(|| sema.parse(file_id).syntax().clone());

        for (idx, _) in text.match_indices(pat) {
            let offset = TextUnit::from_usize(idx);
            if !search_range.contains_inclusive(offset) {
                tested_by!(search_filters_by_range);
                continue;
            }

            let name_ref =
                if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(&tree, offset) {
                    name_ref
                } else {
                    // Handle macro token cases
                    let token = match tree.token_at_offset(offset) {
                        TokenAtOffset::None => continue,
                        TokenAtOffset::Single(t) => t,
                        TokenAtOffset::Between(_, t) => t,
                    };
                    let expanded = sema.descend_into_macros(token);
                    match ast::NameRef::cast(expanded.parent()) {
                        Some(name_ref) => name_ref,
                        _ => continue,
                    }
                };

            if let Some(d) = classify_name_ref(&sema, &name_ref) {
                let d = d.definition();
                if &d == def {
                    let kind =
                        if is_record_lit_name_ref(&name_ref) || is_call_expr_name_ref(&name_ref) {
                            ReferenceKind::StructLiteral
                        } else {
                            ReferenceKind::Other
                        };

                    let file_range = sema.original_range(name_ref.syntax());
                    refs.push(Reference {
                        file_range,
                        kind,
                        access: reference_access(&d, &name_ref),
                    });
                }
            }
        }
    }
    refs
}

fn decl_access(def: &Definition, syntax: &SyntaxNode, range: TextRange) -> Option<ReferenceAccess> {
    match def {
        Definition::Local(_) | Definition::StructField(_) => {}
        _ => return None,
    };

    let stmt = find_node_at_offset::<ast::LetStmt>(syntax, range.start())?;
    if stmt.initializer().is_some() {
        let pat = stmt.pat()?;
        if let ast::Pat::BindPat(it) = pat {
            if it.is_mutable() {
                return Some(ReferenceAccess::Write);
            }
        }
    }

    None
}

fn reference_access(def: &Definition, name_ref: &ast::NameRef) -> Option<ReferenceAccess> {
    // Only Locals and Fields have accesses for now.
    match def {
        Definition::Local(_) | Definition::StructField(_) => {}
        _ => return None,
    };

    let mode = name_ref.syntax().ancestors().find_map(|node| {
        match_ast! {
            match (node) {
                ast::BinExpr(expr) => {
                    if expr.op_kind()?.is_assignment() {
                        // If the variable or field ends on the LHS's end then it's a Write (covers fields and locals).
                        // FIXME: This is not terribly accurate.
                        if let Some(lhs) = expr.lhs() {
                            if lhs.syntax().text_range().end() == name_ref.syntax().text_range().end() {
                                return Some(ReferenceAccess::Write);
                            }
                        }
                    }
                    Some(ReferenceAccess::Read)
                },
                _ => {None}
            }
        }
    });

    // Default Locals and Fields to read
    mode.or(Some(ReferenceAccess::Read))
}

fn is_record_lit_name_ref(name_ref: &ast::NameRef) -> bool {
    name_ref
        .syntax()
        .ancestors()
        .find_map(ast::RecordLit::cast)
        .and_then(|l| l.path())
        .and_then(|p| p.segment())
        .map(|p| p.name_ref().as_ref() == Some(name_ref))
        .unwrap_or(false)
}

fn get_struct_def_name_for_struc_litetal_search(
    syntax: &SyntaxNode,
    position: FilePosition,
) -> Option<ast::Name> {
    if let TokenAtOffset::Between(ref left, ref right) = syntax.token_at_offset(position.offset) {
        if right.kind() != SyntaxKind::L_CURLY && right.kind() != SyntaxKind::L_PAREN {
            return None;
        }
        if let Some(name) = find_node_at_offset::<ast::Name>(&syntax, left.text_range().start()) {
            return name.syntax().ancestors().find_map(ast::StructDef::cast).and_then(|l| l.name());
        }
        if find_node_at_offset::<ast::TypeParamList>(&syntax, left.text_range().start()).is_some() {
            return left.ancestors().find_map(ast::StructDef::cast).and_then(|l| l.name());
        }
    }
    None
}

fn is_call_expr_name_ref(name_ref: &ast::NameRef) -> bool {
    name_ref
        .syntax()
        .ancestors()
        .find_map(ast::CallExpr::cast)
        .and_then(|c| match c.expr()? {
            ast::Expr::PathExpr(p) => {
                Some(p.path()?.segment()?.name_ref().as_ref() == Some(name_ref))
            }
            _ => None,
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use test_utils::covers;

    use crate::{
        mock_analysis::{analysis_and_position, single_file_with_position, MockAnalysis},
        Declaration, Reference, ReferenceSearchResult, SearchScope,
    };

    #[test]
    fn test_struct_literal_after_space() {
        let code = r#"
    struct Foo <|>{
        a: i32,
    }
    impl Foo {
        fn f() -> i32 { 42 }
    }
    fn main() {
        let f: Foo;
        f = Foo {a: Foo::f()};
    }"#;

        let refs = get_all_refs(code);
        check_result(
            refs,
            "Foo STRUCT_DEF FileId(1) [5; 39) [12; 15) Other",
            &["FileId(1) [138; 141) StructLiteral"],
        );
    }

    #[test]
    fn test_struct_literal_befor_space() {
        let code = r#"
    struct Foo<|> {}
        fn main() {
        let f: Foo;
        f = Foo {};
    }"#;

        let refs = get_all_refs(code);
        check_result(
            refs,
            "Foo STRUCT_DEF FileId(1) [5; 18) [12; 15) Other",
            &["FileId(1) [54; 57) Other", "FileId(1) [71; 74) StructLiteral"],
        );
    }

    #[test]
    fn test_struct_literal_with_generic_type() {
        let code = r#"
    struct Foo<T> <|>{}
        fn main() {
        let f: Foo::<i32>;
        f = Foo {};
    }"#;

        let refs = get_all_refs(code);
        check_result(
            refs,
            "Foo STRUCT_DEF FileId(1) [5; 21) [12; 15) Other",
            &["FileId(1) [81; 84) StructLiteral"],
        );
    }

    #[test]
    fn test_struct_literal_for_tuple() {
        let code = r#"
    struct Foo<|>(i32);

    fn main() {
        let f: Foo;
        f = Foo(1);
    }"#;

        let refs = get_all_refs(code);
        check_result(
            refs,
            "Foo STRUCT_DEF FileId(1) [5; 21) [12; 15) Other",
            &["FileId(1) [71; 74) StructLiteral"],
        );
    }

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
        check_result(
            refs,
            "i BIND_PAT FileId(1) [33; 34) Other Write",
            &[
                "FileId(1) [67; 68) Other Write",
                "FileId(1) [71; 72) Other Read",
                "FileId(1) [101; 102) Other Write",
                "FileId(1) [127; 128) Other Write",
            ],
        );
    }

    #[test]
    fn search_filters_by_range() {
        covers!(search_filters_by_range);
        let code = r#"
            fn foo() {
                let spam<|> = 92;
                spam + spam
            }
            fn bar() {
                let spam = 92;
                spam + spam
            }
        "#;
        let refs = get_all_refs(code);
        check_result(
            refs,
            "spam BIND_PAT FileId(1) [44; 48) Other",
            &["FileId(1) [71; 75) Other Read", "FileId(1) [78; 82) Other Read"],
        );
    }

    #[test]
    fn test_find_all_refs_for_param_inside() {
        let code = r#"
    fn foo(i : u32) -> u32 {
        i<|>
    }"#;

        let refs = get_all_refs(code);
        check_result(
            refs,
            "i BIND_PAT FileId(1) [12; 13) Other",
            &["FileId(1) [38; 39) Other Read"],
        );
    }

    #[test]
    fn test_find_all_refs_for_fn_param() {
        let code = r#"
    fn foo(i<|> : u32) -> u32 {
        i
    }"#;

        let refs = get_all_refs(code);
        check_result(
            refs,
            "i BIND_PAT FileId(1) [12; 13) Other",
            &["FileId(1) [38; 39) Other Read"],
        );
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
        check_result(
            refs,
            "spam RECORD_FIELD_DEF FileId(1) [66; 79) [70; 74) Other",
            &["FileId(1) [152; 156) Other Read"],
        );
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
        check_result(refs, "f FN_DEF FileId(1) [88; 104) [91; 92) Other", &[]);
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
        check_result(refs, "B ENUM_VARIANT FileId(1) [83; 84) [83; 84) Other", &[]);
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
        check_result(
            refs,
            "Foo STRUCT_DEF FileId(2) [16; 50) [27; 30) Other",
            &["FileId(1) [52; 55) StructLiteral", "FileId(3) [77; 80) StructLiteral"],
        );
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
        check_result(
            refs,
            "foo SOURCE_FILE FileId(2) [0; 35) Other",
            &["FileId(1) [13; 16) Other"],
        );
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
        check_result(
            refs,
            "Foo STRUCT_DEF FileId(3) [0; 41) [18; 21) Other",
            &["FileId(2) [20; 23) Other", "FileId(2) [46; 49) StructLiteral"],
        );
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
        check_result(
            refs,
            "quux FN_DEF FileId(1) [18; 34) [25; 29) Other",
            &["FileId(2) [16; 20) StructLiteral", "FileId(3) [16; 20) StructLiteral"],
        );

        let refs =
            analysis.find_all_refs(pos, Some(SearchScope::single_file(bar))).unwrap().unwrap();
        check_result(
            refs,
            "quux FN_DEF FileId(1) [18; 34) [25; 29) Other",
            &["FileId(3) [16; 20) StructLiteral"],
        );
    }

    #[test]
    fn test_find_all_refs_macro_def() {
        let code = r#"
        #[macro_export]
        macro_rules! m1<|> { () => (()) }

        fn foo() {
            m1();
            m1();
        }"#;

        let refs = get_all_refs(code);
        check_result(
            refs,
            "m1 MACRO_CALL FileId(1) [9; 63) [46; 48) Other",
            &["FileId(1) [96; 98) StructLiteral", "FileId(1) [114; 116) StructLiteral"],
        );
    }

    #[test]
    fn test_basic_highlight_read_write() {
        let code = r#"
        fn foo() {
            let mut i<|> = 0;
            i = i + 1;
        }"#;

        let refs = get_all_refs(code);
        check_result(
            refs,
            "i BIND_PAT FileId(1) [40; 41) Other Write",
            &["FileId(1) [59; 60) Other Write", "FileId(1) [63; 64) Other Read"],
        );
    }

    #[test]
    fn test_basic_highlight_field_read_write() {
        let code = r#"
        struct S {
            f: u32,
        }

        fn foo() {
            let mut s = S{f: 0};
            s.f<|> = 0;
        }"#;

        let refs = get_all_refs(code);
        check_result(
            refs,
            "f RECORD_FIELD_DEF FileId(1) [32; 38) [32; 33) Other",
            &["FileId(1) [96; 97) Other Read", "FileId(1) [117; 118) Other Write"],
        );
    }

    #[test]
    fn test_basic_highlight_decl_no_write() {
        let code = r#"
        fn foo() {
            let i<|>;
            i = 1;
        }"#;

        let refs = get_all_refs(code);
        check_result(
            refs,
            "i BIND_PAT FileId(1) [36; 37) Other",
            &["FileId(1) [51; 52) Other Write"],
        );
    }

    fn get_all_refs(text: &str) -> ReferenceSearchResult {
        let (analysis, position) = single_file_with_position(text);
        analysis.find_all_refs(position, None).unwrap().unwrap()
    }

    fn check_result(res: ReferenceSearchResult, expected_decl: &str, expected_refs: &[&str]) {
        res.declaration().assert_match(expected_decl);
        assert_eq!(res.references.len(), expected_refs.len());
        res.references().iter().enumerate().for_each(|(i, r)| r.assert_match(expected_refs[i]));
    }

    impl Declaration {
        fn debug_render(&self) -> String {
            let mut s = format!("{} {:?}", self.nav.debug_render(), self.kind);
            if let Some(access) = self.access {
                s.push_str(&format!(" {:?}", access));
            }
            s
        }

        fn assert_match(&self, expected: &str) {
            let actual = self.debug_render();
            test_utils::assert_eq_text!(expected.trim(), actual.trim(),);
        }
    }

    impl Reference {
        fn debug_render(&self) -> String {
            let mut s = format!(
                "{:?} {:?} {:?}",
                self.file_range.file_id, self.file_range.range, self.kind
            );
            if let Some(access) = self.access {
                s.push_str(&format!(" {:?}", access));
            }
            s
        }

        fn assert_match(&self, expected: &str) {
            let actual = self.debug_render();
            test_utils::assert_eq_text!(expected.trim(), actual.trim(),);
        }
    }
}
