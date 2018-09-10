use std::sync::Arc;
use {
    FileId,
    db::{Query, Eval, QueryCtx, FileSyntax, Files},
    module_map::resolve_submodule,
};

enum ModuleDescr {}
impl Query for ModuleDescr {
    const ID: u32 = 30;
    type Params = FileId;
    type Output = Arc<descr::ModuleDescr>;
}

enum ResolveSubmodule {}
impl Query for ResolveSubmodule {
    const ID: u32 = 31;
    type Params = (FileId, descr::Submodule);
    type Output = Arc<Vec<FileId>>;
}

enum ParentModule {}
impl Query for ParentModule {
    const ID: u32 = 40;
    type Params = FileId;
    type Output = Arc<Vec<FileId>>;
}

impl Eval for ModuleDescr {
    fn eval(ctx: &QueryCtx, file_id: &FileId) -> Arc<descr::ModuleDescr> {
        let file = ctx.get::<FileSyntax>(file_id);
        Arc::new(descr::ModuleDescr::new(file.ast()))
    }
}

impl Eval for ResolveSubmodule {
    fn eval(ctx: &QueryCtx, &(file_id, ref submodule): &(FileId, descr::Submodule)) -> Arc<Vec<FileId>> {
        let files = ctx.get::<Files>(&());
        let res = resolve_submodule(file_id, &submodule.name, &files.file_resolver()).0;
        Arc::new(res)
    }
}

impl Eval for ParentModule {
    fn eval(ctx: &QueryCtx, file_id: &FileId) -> Arc<Vec<FileId>> {
        let files = ctx.get::<Files>(&());
        let res = files.iter()
            .map(|parent_id| (parent_id, ctx.get::<ModuleDescr>(&parent_id)))
            .filter(|(parent_id, descr)| {
                descr.submodules.iter()
                    .any(|subm| {
                        ctx.get::<ResolveSubmodule>(&(*parent_id, subm.clone()))
                            .iter()
                            .any(|it| it == file_id)
                    })
            })
            .map(|(id, _)| id)
            .collect();
        Arc::new(res)
    }
}

mod descr {
    use libsyntax2::{
        SmolStr,
        ast::{self, NameOwner},
    };

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

    #[derive(Clone, Hash)]
    pub struct Submodule {
        pub name: SmolStr,
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use im;
    use relative_path::{RelativePath, RelativePathBuf};
    use {
        db::Db,
        imp::FileResolverImp,
        FileId, FileResolver,
    };

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
                db: Db::new(),
            }
        }
        fn add_file(&mut self, path: &str, text: &str) -> FileId {
            assert!(path.starts_with("/"));
            let file_id = FileId(self.next_file_id);
            self.next_file_id += 1;
            self.fm.insert(file_id, RelativePathBuf::from(&path[1..]));
            self.db.change_file(file_id, Some(text.to_string()));
            self.db.set_file_resolver(FileResolverImp::new(
                Arc::new(FileMap(self.fm.clone()))
            ));

            file_id
        }
        fn remove_file(&mut self, file_id: FileId) {
            self.fm.remove(&file_id);
            self.db.change_file(file_id, None);
            self.db.set_file_resolver(FileResolverImp::new(
                Arc::new(FileMap(self.fm.clone()))
            ))
        }
        fn change_file(&mut self, file_id: FileId, new_text: &str) {
            self.db.change_file(file_id, Some(new_text.to_string()));
        }
        fn check_parent_modules(&self, file_id: FileId, expected: &[FileId]) {
            let ctx = self.db.query_ctx();
            let actual = ctx.get::<ParentModule>(&file_id);
            assert_eq!(actual.as_slice(), expected);
        }
    }

    #[test]
    fn test_parent_module() {
        let mut f = Fixture::new();
        let foo = f.add_file("/foo.rs", "");
        f.check_parent_modules(foo, &[]);

        let lib = f.add_file("/lib.rs", "mod foo;");
        f.check_parent_modules(foo, &[lib]);

        f.change_file(lib, "");
        f.check_parent_modules(foo, &[]);

        f.change_file(lib, "mod foo;");
        f.check_parent_modules(foo, &[lib]);

        f.change_file(lib, "mod bar;");
        f.check_parent_modules(foo, &[]);

        f.change_file(lib, "mod foo;");
        f.check_parent_modules(foo, &[lib]);

        f.remove_file(lib);
        f.check_parent_modules(foo, &[]);
    }

}
