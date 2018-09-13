use std::sync::Arc;
use {
    FileId,
    db::{
        Query, QueryCtx
    },
    module_map::resolve_submodule,
};

impl<'a> QueryCtx<'a> {
    fn module_descr(&self, file_id: FileId) -> Arc<descr::ModuleDescr> {
        self.get(MODULE_DESCR, file_id)
    }
    fn resolve_submodule(&self, file_id: FileId, submod: descr::Submodule) -> Arc<Vec<FileId>> {
        self.get(RESOLVE_SUBMODULE, (file_id, submod))
    }
}

pub(crate) const MODULE_DESCR: Query<FileId, descr::ModuleDescr> = Query {
    id: 30,
    f: |ctx, &file_id| {
        let file = ctx.file_syntax(file_id);
        descr::ModuleDescr::new(file.ast())
    }
};

pub(crate) const RESOLVE_SUBMODULE: Query<(FileId, descr::Submodule), Vec<FileId>> = Query {
    id: 31,
    f: |ctx, params| {
        let files = ctx.file_set();
        resolve_submodule(params.0, &params.1.name, &files.1).0
    }
};

pub(crate) const PARENT_MODULE: Query<FileId, Vec<FileId>> = Query {
    id: 40,
    f: |ctx, file_id| {
        let files = ctx.file_set();
        let res = files.0.iter()
            .map(|&parent_id| (parent_id, ctx.module_descr(parent_id)))
            .filter(|(parent_id, descr)| {
                descr.submodules.iter()
                    .any(|subm| {
                        ctx.resolve_submodule(*parent_id, subm.clone())
                            .iter()
                            .any(|it| it == file_id)
                    })
            })
            .map(|(id, _)| id)
            .collect();
        res
    }
};

mod descr {
    use libsyntax2::{
        SmolStr,
        ast::{self, NameOwner},
    };

    #[derive(Debug, Hash)]
    pub struct ModuleDescr {
        pub submodules: Vec<Submodule>
    }

    impl ModuleDescr {
        pub fn new(root: ast::Root) -> ModuleDescr {
            let submodules = root
                .modules()
                .filter_map(|module| {
                    let name = module.name()?.text();
                    if !module.has_semi() {
                        return None;
                    }
                    Some(Submodule { name })
                }).collect();

            ModuleDescr { submodules } }
    }

    #[derive(Clone, Hash, PartialEq, Eq, Debug)]
    pub struct Submodule {
        pub name: SmolStr,
    }

}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use im;
    use relative_path::{RelativePath, RelativePathBuf};
    use {
        db::{Query, Db, State},
        imp::FileResolverImp,
        FileId, FileResolver,
    };
    use super::*;

    #[derive(Debug)]
    struct FileMap(im::HashMap<FileId, RelativePathBuf>);

    impl FileResolver for FileMap {
        fn file_stem(&self, file_id: FileId) -> String {
            self.0[&file_id].file_stem().unwrap().to_string()
        }
        fn resolve(&self, file_id: FileId, rel: &RelativePath) -> Option<FileId> {
            let path = self.0[&file_id].join(rel).normalize();
            self.0.iter()
                .filter_map(|&(id, ref p)| Some(id).filter(|_| p == &path))
                .next()
        }
    }

    struct Fixture {
        next_file_id: u32,
        fm: im::HashMap<FileId, RelativePathBuf>,
        db: Db,
    }

    impl Fixture {
        fn new() -> Fixture {
            Fixture {
                next_file_id: 1,
                fm: im::HashMap::new(),
                db: Db::new(State::default()),
            }
        }
        fn add_file(&mut self, path: &str, text: &str) -> FileId {
            assert!(path.starts_with("/"));
            let file_id = FileId(self.next_file_id);
            self.next_file_id += 1;
            self.fm.insert(file_id, RelativePathBuf::from(&path[1..]));
            let mut new_state = self.db.state().clone();
            new_state.file_map.insert(file_id, text.to_string().into_boxed_str().into());
            new_state.resolver = FileResolverImp::new(
                Arc::new(FileMap(self.fm.clone()))
            );
            self.db = self.db.with_state(new_state, &[file_id], true);
            file_id
        }
        fn remove_file(&mut self, file_id: FileId) {
            self.fm.remove(&file_id);
            let mut new_state = self.db.state().clone();
            new_state.file_map.remove(&file_id);
            new_state.resolver = FileResolverImp::new(
                Arc::new(FileMap(self.fm.clone()))
            );
            self.db = self.db.with_state(new_state, &[file_id], true);
        }
        fn change_file(&mut self, file_id: FileId, new_text: &str) {
            let mut new_state = self.db.state().clone();
            new_state.file_map.insert(file_id, new_text.to_string().into_boxed_str().into());
            self.db = self.db.with_state(new_state, &[file_id], false);
        }
        fn check_parent_modules(
            &self,
            file_id: FileId,
            expected: &[FileId],
            queries: &[(u16, u64)]
        ) {
            let (actual, events) = self.db.get(PARENT_MODULE, file_id);
            assert_eq!(actual.as_slice(), expected);
            let mut counts = HashMap::new();
            events.into_iter()
               .for_each(|event| *counts.entry(event).or_insert(0) += 1);
            for &(query_id, expected_count) in queries.iter() {
                let actual_count = *counts.get(&query_id).unwrap_or(&0);
                assert_eq!(
                    actual_count,
                    expected_count,
                    "counts for {} differ",
                    query_id,
                )
            }

        }
    }

    #[test]
    fn test_parent_module() {
        let mut f = Fixture::new();
        let foo = f.add_file("/foo.rs", "");
        f.check_parent_modules(foo, &[], &[(MODULE_DESCR.id, 1)]);

        let lib = f.add_file("/lib.rs", "mod foo;");
        f.check_parent_modules(foo, &[lib], &[(MODULE_DESCR.id, 1)]);
        f.check_parent_modules(foo, &[lib], &[(MODULE_DESCR.id, 0)]);

        f.change_file(lib, "");
        f.check_parent_modules(foo, &[], &[(MODULE_DESCR.id, 1)]);

        f.change_file(lib, "mod foo;");
        f.check_parent_modules(foo, &[lib], &[(MODULE_DESCR.id, 1)]);

        f.change_file(lib, "mod bar;");
        f.check_parent_modules(foo, &[], &[(MODULE_DESCR.id, 1)]);

        f.change_file(lib, "mod foo;");
        f.check_parent_modules(foo, &[lib], &[(MODULE_DESCR.id, 1)]);

        f.remove_file(lib);
        f.check_parent_modules(foo, &[], &[(MODULE_DESCR.id, 0)]);
    }
}
