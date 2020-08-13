//! This module resolves `mod foo;` declaration to file.
use base_db::FileId;
use hir_expand::name::Name;
use syntax::SmolStr;

use crate::{db::DefDatabase, HirFileId};

#[derive(Clone, Debug)]
pub(super) struct ModDir {
    /// `` for `mod.rs`, `lib.rs`
    /// `foo/` for `foo.rs`
    /// `foo/bar/` for `mod bar { mod x; }` nested in `foo.rs`
    /// Invariant: path.is_empty() || path.ends_with('/')
    dir_path: DirPath,
    /// inside `./foo.rs`, mods with `#[path]` should *not* be relative to `./foo/`
    root_non_dir_owner: bool,
}

impl ModDir {
    pub(super) fn root() -> ModDir {
        ModDir { dir_path: DirPath::empty(), root_non_dir_owner: false }
    }

    pub(super) fn descend_into_definition(
        &self,
        name: &Name,
        attr_path: Option<&SmolStr>,
    ) -> ModDir {
        let path = match attr_path.map(|it| it.as_str()) {
            None => {
                let mut path = self.dir_path.clone();
                path.push(&name.to_string());
                path
            }
            Some(attr_path) => {
                let mut path = self.dir_path.join_attr(attr_path, self.root_non_dir_owner);
                if !(path.is_empty() || path.ends_with('/')) {
                    path.push('/')
                }
                DirPath::new(path)
            }
        };
        ModDir { dir_path: path, root_non_dir_owner: false }
    }

    pub(super) fn resolve_declaration(
        &self,
        db: &dyn DefDatabase,
        file_id: HirFileId,
        name: &Name,
        attr_path: Option<&SmolStr>,
    ) -> Result<(FileId, bool, ModDir), String> {
        let file_id = file_id.original_file(db.upcast());

        let mut candidate_files = Vec::new();
        match attr_path {
            Some(attr_path) => {
                candidate_files.push(self.dir_path.join_attr(attr_path, self.root_non_dir_owner))
            }
            None => {
                candidate_files.push(format!("{}{}.rs", self.dir_path.0, name));
                candidate_files.push(format!("{}{}/mod.rs", self.dir_path.0, name));
            }
        };

        for candidate in candidate_files.iter() {
            if let Some(file_id) = db.resolve_path(file_id, candidate.as_str()) {
                let is_mod_rs = candidate.ends_with("mod.rs");

                let (dir_path, root_non_dir_owner) = if is_mod_rs || attr_path.is_some() {
                    (DirPath::empty(), false)
                } else {
                    (DirPath::new(format!("{}/", name)), true)
                };
                return Ok((file_id, is_mod_rs, ModDir { dir_path, root_non_dir_owner }));
            }
        }
        Err(candidate_files.remove(0))
    }
}

#[derive(Clone, Debug)]
struct DirPath(String);

impl DirPath {
    fn assert_invariant(&self) {
        assert!(self.0.is_empty() || self.0.ends_with('/'));
    }
    fn new(repr: String) -> DirPath {
        let res = DirPath(repr);
        res.assert_invariant();
        res
    }
    fn empty() -> DirPath {
        DirPath::new(String::new())
    }
    fn push(&mut self, name: &str) {
        self.0.push_str(name);
        self.0.push('/');
        self.assert_invariant();
    }
    fn parent(&self) -> Option<&str> {
        if self.0.is_empty() {
            return None;
        };
        let idx =
            self.0[..self.0.len() - '/'.len_utf8()].rfind('/').map_or(0, |it| it + '/'.len_utf8());
        Some(&self.0[..idx])
    }
    /// So this is the case which doesn't really work I think if we try to be
    /// 100% platform agnostic:
    ///
    /// ```
    /// mod a {
    ///     #[path="C://sad/face"]
    ///     mod b { mod c; }
    /// }
    /// ```
    ///
    /// Here, we need to join logical dir path to a string path from an
    /// attribute. Ideally, we should somehow losslessly communicate the whole
    /// construction to `FileLoader`.
    fn join_attr(&self, mut attr: &str, relative_to_parent: bool) -> String {
        let base = if relative_to_parent { self.parent().unwrap() } else { &self.0 };

        if attr.starts_with("./") {
            attr = &attr["./".len()..];
        }
        let tmp;
        let attr = if attr.contains('\\') {
            tmp = attr.replace('\\', "/");
            &tmp
        } else {
            attr
        };
        let res = format!("{}{}", base, attr);
        res
    }
}
