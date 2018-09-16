use std::sync::Arc;
use {
    FileId,
    db::{
        Query, QueryRegistry, QueryCtx,
        file_set
    },
    queries::file_syntax,
    descriptors::{ModuleDescriptor, ModuleTreeDescriptor},
};

pub(crate) fn register_queries(reg: &mut QueryRegistry) {
    reg.add(MODULE_DESCR, "MODULE_DESCR");
    reg.add(MODULE_TREE, "MODULE_TREE");
}

pub(crate) fn module_tree(ctx: QueryCtx) -> Arc<ModuleTreeDescriptor> {
    ctx.get(MODULE_TREE, ())
}

const MODULE_DESCR: Query<FileId, ModuleDescriptor> = Query(30, |ctx, &file_id| {
    let file = file_syntax(ctx, file_id);
    ModuleDescriptor::new(file.ast())
});

const MODULE_TREE: Query<(), ModuleTreeDescriptor> = Query(31, |ctx, _| {
    let file_set = file_set(ctx);
    let mut files = Vec::new();
    for &file_id in file_set.0.iter() {
        let module_descr = ctx.get(MODULE_DESCR, file_id);
        files.push((file_id, module_descr));
    }
    ModuleTreeDescriptor::new(files.iter().map(|(file_id, descr)| (*file_id, &**descr)), &file_set.1)
});

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use im;
    use relative_path::{RelativePath, RelativePathBuf};
    use {
        db::{Db},
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
                db: Db::new(),
            }
        }
        fn add_file(&mut self, path: &str, text: &str) -> FileId {
            assert!(path.starts_with("/"));
            let file_id = FileId(self.next_file_id);
            self.next_file_id += 1;
            self.fm.insert(file_id, RelativePathBuf::from(&path[1..]));
            let mut new_state = self.db.state().clone();
            new_state.file_map.insert(file_id, Arc::new(text.to_string()));
            new_state.file_resolver = FileResolverImp::new(
                Arc::new(FileMap(self.fm.clone()))
            );
            self.db = self.db.with_changes(new_state, &[file_id], true);
            file_id
        }
        fn remove_file(&mut self, file_id: FileId) {
            self.fm.remove(&file_id);
            let mut new_state = self.db.state().clone();
            new_state.file_map.remove(&file_id);
            new_state.file_resolver = FileResolverImp::new(
                Arc::new(FileMap(self.fm.clone()))
            );
            self.db = self.db.with_changes(new_state, &[file_id], true);
        }
        fn change_file(&mut self, file_id: FileId, new_text: &str) {
            let mut new_state = self.db.state().clone();
            new_state.file_map.insert(file_id, Arc::new(new_text.to_string()));
            self.db = self.db.with_changes(new_state, &[file_id], false);
        }
        fn check_parent_modules(
            &self,
            file_id: FileId,
            expected: &[FileId],
            queries: &[(&'static str, u64)]
        ) {
            let (tree, events) = self.db.trace_query(|ctx| module_tree(ctx));
            let actual = tree.parent_modules(file_id)
                .into_iter()
                .map(|link| link.owner(&tree))
                .collect::<Vec<_>>();
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
        f.check_parent_modules(foo, &[], &[("MODULE_DESCR", 1)]);

        let lib = f.add_file("/lib.rs", "mod foo;");
        f.check_parent_modules(foo, &[lib], &[("MODULE_DESCR", 1)]);
        f.check_parent_modules(foo, &[lib], &[("MODULE_DESCR", 0)]);

        f.change_file(lib, "");
        f.check_parent_modules(foo, &[], &[("MODULE_DESCR", 1)]);

        f.change_file(lib, "mod foo;");
        f.check_parent_modules(foo, &[lib], &[("MODULE_DESCR", 1)]);

        f.change_file(lib, "mod bar;");
        f.check_parent_modules(foo, &[], &[("MODULE_DESCR", 1)]);

        f.change_file(lib, "mod foo;");
        f.check_parent_modules(foo, &[lib], &[("MODULE_DESCR", 1)]);

        f.remove_file(lib);
        f.check_parent_modules(foo, &[], &[("MODULE_DESCR", 0)]);
    }
}
