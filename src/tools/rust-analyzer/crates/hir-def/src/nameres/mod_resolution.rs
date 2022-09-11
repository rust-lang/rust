//! This module resolves `mod foo;` declaration to file.
use arrayvec::ArrayVec;
use base_db::{AnchoredPath, FileId};
use hir_expand::name::Name;
use limit::Limit;
use syntax::SmolStr;

use crate::{db::DefDatabase, HirFileId};

const MOD_DEPTH_LIMIT: Limit = Limit::new(32);

#[derive(Clone, Debug)]
pub(super) struct ModDir {
    /// `` for `mod.rs`, `lib.rs`
    /// `foo/` for `foo.rs`
    /// `foo/bar/` for `mod bar { mod x; }` nested in `foo.rs`
    /// Invariant: path.is_empty() || path.ends_with('/')
    dir_path: DirPath,
    /// inside `./foo.rs`, mods with `#[path]` should *not* be relative to `./foo/`
    root_non_dir_owner: bool,
    depth: u32,
}

impl ModDir {
    pub(super) fn root() -> ModDir {
        ModDir { dir_path: DirPath::empty(), root_non_dir_owner: false, depth: 0 }
    }

    pub(super) fn descend_into_definition(
        &self,
        name: &Name,
        attr_path: Option<&SmolStr>,
    ) -> Option<ModDir> {
        let path = match attr_path.map(SmolStr::as_str) {
            None => {
                let mut path = self.dir_path.clone();
                path.push(&name.to_smol_str());
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
        self.child(path, false)
    }

    fn child(&self, dir_path: DirPath, root_non_dir_owner: bool) -> Option<ModDir> {
        let depth = self.depth + 1;
        if MOD_DEPTH_LIMIT.check(depth as usize).is_err() {
            tracing::error!("MOD_DEPTH_LIMIT exceeded");
            cov_mark::hit!(circular_mods);
            return None;
        }
        Some(ModDir { dir_path, root_non_dir_owner, depth })
    }

    pub(super) fn resolve_declaration(
        &self,
        db: &dyn DefDatabase,
        file_id: HirFileId,
        name: &Name,
        attr_path: Option<&SmolStr>,
    ) -> Result<(FileId, bool, ModDir), Box<[String]>> {
        let name = name.unescaped();
        let orig_file_id = file_id.original_file(db.upcast());

        let mut candidate_files = ArrayVec::<_, 2>::new();
        match attr_path {
            Some(attr_path) => {
                candidate_files.push(self.dir_path.join_attr(attr_path, self.root_non_dir_owner))
            }
            None if file_id.is_include_macro(db.upcast()) => {
                candidate_files.push(format!("{}.rs", name));
                candidate_files.push(format!("{}/mod.rs", name));
            }
            None => {
                candidate_files.push(format!("{}{}.rs", self.dir_path.0, name));
                candidate_files.push(format!("{}{}/mod.rs", self.dir_path.0, name));
            }
        };

        for candidate in candidate_files.iter() {
            let path = AnchoredPath { anchor: orig_file_id, path: candidate.as_str() };
            if let Some(file_id) = db.resolve_path(path) {
                let is_mod_rs = candidate.ends_with("/mod.rs");

                let (dir_path, root_non_dir_owner) = if is_mod_rs || attr_path.is_some() {
                    (DirPath::empty(), false)
                } else {
                    (DirPath::new(format!("{}/", name)), true)
                };
                if let Some(mod_dir) = self.child(dir_path, root_non_dir_owner) {
                    return Ok((file_id, is_mod_rs, mod_dir));
                }
            }
        }
        Err(candidate_files.into_iter().collect())
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
