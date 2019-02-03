mod fill_match_arm;

use ra_syntax::{
    TextRange, SourceFile, AstNode,
    algo::find_node_at_offset,
};
use ra_ide_api_light::{
    LocalEdit,
    assists::{
        Assist,
        AssistBuilder
    }
};
use crate::{
    db::RootDatabase,
    FileId
};

/// Return all the assists applicable at the given position.
pub(crate) fn assists(
    db: &RootDatabase,
    file_id: FileId,
    file: &SourceFile,
    range: TextRange,
) -> Vec<LocalEdit> {
    let ctx = AssistCtx::new(db, file_id, file, range);
    [fill_match_arm::fill_match_arm]
        .iter()
        .filter_map(|&assist| ctx.clone().apply(assist))
        .collect()
}

#[derive(Debug, Clone)]
pub struct AssistCtx<'a> {
    file_id: FileId,
    source_file: &'a SourceFile,
    db: &'a RootDatabase,
    range: TextRange,
    should_compute_edit: bool,
}

impl<'a> AssistCtx<'a> {
    pub(crate) fn new(
        db: &'a RootDatabase,
        file_id: FileId,
        source_file: &'a SourceFile,
        range: TextRange,
    ) -> AssistCtx<'a> {
        AssistCtx {
            source_file,
            file_id,
            db,
            range,
            should_compute_edit: false,
        }
    }

    pub fn apply(mut self, assist: fn(AssistCtx) -> Option<Assist>) -> Option<LocalEdit> {
        self.should_compute_edit = true;
        match assist(self) {
            None => None,
            Some(Assist::Edit(e)) => Some(e),
            Some(Assist::Applicable) => unreachable!(),
        }
    }

    #[allow(unused)]
    pub fn check(mut self, assist: fn(AssistCtx) -> Option<Assist>) -> bool {
        self.should_compute_edit = false;
        match assist(self) {
            None => false,
            Some(Assist::Edit(_)) => unreachable!(),
            Some(Assist::Applicable) => true,
        }
    }

    fn build(self, label: impl Into<String>, f: impl FnOnce(&mut AssistBuilder)) -> Option<Assist> {
        if !self.should_compute_edit {
            return Some(Assist::Applicable);
        }
        let mut edit = AssistBuilder::default();
        f(&mut edit);
        Some(edit.build(label))
    }

    pub(crate) fn node_at_offset<N: AstNode>(&self) -> Option<&'a N> {
        find_node_at_offset(self.source_file.syntax(), self.range.start())
    }
}
