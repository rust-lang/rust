use std::sync::Arc;
use libsyntax2::{File};
use {
    FileId, FileResolverImp,
    db::{Query, GroundQuery, QueryCtx, hash},
};


impl<'a> QueryCtx<'a> {
    pub(crate) fn file_set(&self) -> Arc<(Vec<FileId>, FileResolverImp)> {
        self.get_g(FILE_SET, ())
    }
    pub(crate) fn file_text(&self, file_id: FileId) -> Arc<str> {
        Arc::clone(&*self.get_g(FILE_TEXT, file_id))
    }
    pub(crate) fn file_syntax(&self, file_id: FileId) -> File {
        (&*self.get(FILE_SYNTAX, file_id)).clone()
    }
}

pub(super) const FILE_TEXT: GroundQuery<FileId, Arc<str>> = GroundQuery {
    id: 10,
    f: |state, id| state.file_map[&id].clone(),
    h: hash,
};

pub(super) const FILE_SET: GroundQuery<(), (Vec<FileId>, FileResolverImp)> = GroundQuery {
    id: 11,
    f: |state, &()| {
        let files = state.file_map.keys().cloned().collect();
        let resolver = state.resolver.clone();
        (files, resolver)
    },
    h: |(files, _)| hash(files),
};

pub(super) const FILE_SYNTAX: Query<FileId, File> = Query {
    id: 20,
    f: |ctx, file_id: &FileId| {
        let text = ctx.file_text(*file_id);
        File::parse(&*text)
    }
};
