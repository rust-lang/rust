pub enum SearchScope {
    Function(hir::Function),
    Module(hir::Module),
    Crate(hir::Crate),
    Crates(Vec<hir::Crate>),
}

pub struct SearchScope{ 
    pub scope: Vec<SyntaxNode>
}

pub fn find_all_refs(db: &RootDatabase, decl: NameKind) -> Vec<ReferenceDescriptor> {
    let (module, visibility) = match decl {
        FieldAccess(field) => {
            let parent = field.parent_def(db);
            let module = parent.module(db);
            let visibility = match parent {
                VariantDef::Struct(s) => s.source(db).ast.visibility(),
                VariantDef::EnumVariant(v) => v.parent_enum(db).source(db).ast.visibility(),
            };
            (module, visibility)
        }
        AssocItem(item) => {
            let parent = item.parent_trait(db)?;
            let module = parent.module(db);
            let visibility = parent.source(db).ast.visibility();
            (module, visibility)
        }
        Def(def) => {
            let (module, visibility) = match def {
                ModuleDef::Module(m) => (m, ),
                ModuleDef::Function(f) => (f.module(db), f.source(db).ast.visibility()),
                ModuleDef::Adt::Struct(s) => (s.module(db), s.source(db).ast.visibility()),
                ModuleDef::Adt::Union(u) => (u.module(db), u.source(db).ast.visibility()),
                ModuleDef::Adt::Enum(e) => (e.module(db), e.source(db).ast.visibility()),
                ModuleDef::EnumVariant(v) => (v.module(db), v.source(db).ast.visibility()),
                ModuleDef::Const(c) => (c.module(db), c.source(db).ast.visibility()),
                ModuleDef::Static(s) => (s.module(db), s.source(db).ast.visibility()),
                ModuleDef::Trait(t) => (t.module(db), t.source(db).ast.visibility()),
                ModuleDef::TypeAlias(a) => (a.module(db), a.source(db).ast.visibility()),
                ModuleDef::BuiltinType(_) => return vec![];
            };
            (module, visibility)
        }
        // FIXME: add missing kinds
        _ => return vec![];
    };
    let scope = scope(db, module, visibility);
}

fn scope(db: &RootDatabase, module: hir::Module, item_vis: Option<ast::Visibility>) -> SearchScope {
    if let Some(v) = item_vis {
        let krate = module.krate(db)?;

        if v.syntax().text() == "pub" {
            SearchScope::Crate(krate)
        }
        if v.syntax().text() == "pub(crate)" {
            let crate_graph = db.crate_graph();
            let crates = crate_graph.iter().filter(|id| {
                crate_graph.dependencies(id).any(|d| d.crate_id() == krate.crate_id())
            }).map(|id| Crate { id }).collect::<Vec<_>>();
            crates.insert(0, krate);
            SearchScope::Crates(crates)
        }
        // FIXME: "pub(super)", "pub(in path)"
        SearchScope::Module(module)
    }
    SearchScope::Module(module)
}

fn process_one(db, scope: SearchScope, pat) {
    match scope {
        SearchScope::Crate(krate) => {
            let text = db.file_text(position.file_id).as_str();
            let parse = SourceFile::parse(text);
            for (offset, name) in text.match_indices(pat) {
                if let Some() = find_node_at_offset<ast::NameRef>(parse, offset) {
                    
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        mock_analysis::analysis_and_position, mock_analysis::single_file_with_position, FileId,
        ReferenceSearchResult,
    };
    use insta::assert_debug_snapshot;
    use test_utils::assert_eq_text;

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
            }
        "#;

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
                spam<|>: u32,
            }
        "#;

        let refs = get_all_refs(code);
        assert_eq!(refs.len(), 1);
    }

    #[test]
    fn test_find_all_refs_methods() {
        let code = r#"
            //- /lib.rs
            struct Foo;
            impl Foo {
                pub fn a() {
                    self.b()
                }
                fn b(&self) {}
            }
        "#;

        let refs = get_all_refs(code);
        assert_eq!(refs.len(), 1);
    }

    #[test]
    fn test_find_all_refs_pub_enum() {
        let code = r#"
            //- /lib.rs
            pub enum Foo {
                A,
                B<|>,
                C,
            }
        "#;

        let refs = get_all_refs(code);
        assert_eq!(refs.len(), 1);
    }

    fn get_all_refs(text: &str) -> ReferenceSearchResult {
        let (analysis, position) = single_file_with_position(text);
        analysis.find_all_refs(position).unwrap().unwrap()
    }
}