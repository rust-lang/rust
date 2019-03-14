use ra_syntax::{
    AstNode, SourceFile, TreeArc, AstPtr,
    ast,
};
use ra_arena::{RawId, impl_arena_id, map::ArenaMap};

use crate::{Path, ModuleSource, Name};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImportId(RawId);
impl_arena_id!(ImportId);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportData {
    pub(super) path: Path,
    pub(super) alias: Option<Name>,
    pub(super) is_glob: bool,
    pub(super) is_prelude: bool,
    pub(super) is_extern_crate: bool,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ImportSourceMap {
    map: ArenaMap<ImportId, AstPtr<ast::PathSegment>>,
}

impl ImportSourceMap {
    pub(crate) fn insert(&mut self, import: ImportId, segment: &ast::PathSegment) {
        self.map.insert(import, AstPtr::new(segment))
    }

    pub fn get(&self, source: &ModuleSource, import: ImportId) -> TreeArc<ast::PathSegment> {
        let file = match source {
            ModuleSource::SourceFile(file) => &*file,
            ModuleSource::Module(m) => m.syntax().ancestors().find_map(SourceFile::cast).unwrap(),
        };

        self.map[import].to_node(file).to_owned()
    }
}
