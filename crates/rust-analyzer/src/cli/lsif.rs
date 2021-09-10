//! Lsif generator

use std::env;

use ide::{StaticIndex, StaticIndexedFile, TokenStaticData};
use ide_db::LineIndexDatabase;

use ide_db::base_db::salsa::{self, ParallelDatabase};
use lsp_types::{Hover, HoverContents, NumberOrString};
use project_model::{CargoConfig, ProjectManifest, ProjectWorkspace};
use vfs::AbsPathBuf;

use crate::cli::lsif::lsif_types::{Document, Vertex};
use crate::cli::{
    flags,
    load_cargo::{load_workspace, LoadCargoConfig},
    Result,
};
use crate::line_index::{LineEndings, LineIndex, OffsetEncoding};
use crate::to_proto;

/// Need to wrap Snapshot to provide `Clone` impl for `map_with`
struct Snap<DB>(DB);
impl<DB: ParallelDatabase> Clone for Snap<salsa::Snapshot<DB>> {
    fn clone(&self) -> Snap<salsa::Snapshot<DB>> {
        Snap(self.0.snapshot())
    }
}

mod lsif_types;
use lsif_types::*;

#[derive(Default)]
struct LsifManager {
    count: i32,
}

#[derive(Clone, Copy)]
struct Id(i32);

impl From<Id> for NumberOrString {
    fn from(Id(x): Id) -> Self {
        NumberOrString::Number(x)
    }
}

impl LsifManager {
    fn add(&mut self, data: Element) -> Id {
        let id = Id(self.count);
        self.emit(&serde_json::to_string(&Entry { id: id.into(), data }).unwrap());
        self.count += 1;
        id
    }

    // FIXME: support file in addition to stdout here
    fn emit(&self, data: &str) {
        println!("{}", data);
    }

    fn add_tokens(
        &mut self,
        line_index: &LineIndex,
        doc_id: Id,
        tokens: Vec<TokenStaticData>,
    ) {
        let tokens_id = tokens
            .into_iter()
            .map(|token| {
                let token_id = self
                    .add(Element::Vertex(Vertex::Range(to_proto::range(line_index, token.range))));
                if let Some(hover) = token.hover {
                    let hover_id = self.add(Element::Vertex(Vertex::HoverResult {
                        result: Hover {
                            contents: HoverContents::Markup(to_proto::markup_content(hover.markup)),
                            range: None,
                        },
                    }));
                    self.add(Element::Edge(Edge::Hover(EdgeData {
                        in_v: hover_id.into(),
                        out_v: token_id.into(),
                    })));
                }
                token_id.into()
            })
            .collect();
        self.add(Element::Edge(Edge::Contains(EdgeDataMultiIn {
            in_vs: tokens_id,
            out_v: doc_id.into(),
        })));
    }
}

impl flags::Lsif {
    pub fn run(self) -> Result<()> {
        let cargo_config = CargoConfig::default();
        let no_progress = &|_| ();
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: true,
            with_proc_macro: true,
            prefill_caches: false,
        };
        let path = AbsPathBuf::assert(env::current_dir()?.join(&self.path));
        let manifest = ProjectManifest::discover_single(&path)?;

        let workspace = ProjectWorkspace::load(manifest, &cargo_config, no_progress)?;

        let (host, vfs, _proc_macro) = load_workspace(workspace, &load_cargo_config)?;
        let db = host.raw_database();
        let analysis = host.analysis();

        let si = StaticIndex::compute(db, &analysis)?;

        let mut lsif = LsifManager::default();
        lsif.add(Element::Vertex(Vertex::MetaData {
            version: String::from("0.5.0"),
            project_root: lsp_types::Url::from_file_path(path).unwrap(),
            position_encoding: Encoding::Utf16,
            tool_info: None,
        }));
        for StaticIndexedFile { file_id, folds, tokens } in si.files {
            let path = vfs.file_path(file_id);
            let path = path.as_path().unwrap();
            let doc_id = lsif.add(Element::Vertex(Vertex::Document(Document {
                language_id: Language::Rust,
                uri: lsp_types::Url::from_file_path(path).unwrap(),
            })));
            let text = analysis.file_text(file_id)?;
            let line_index = db.line_index(file_id);
            let line_index = LineIndex {
                index: line_index.clone(),
                encoding: OffsetEncoding::Utf16,
                endings: LineEndings::Unix,
            };
            let result = folds
                .into_iter()
                .map(|it| to_proto::folding_range(&*text, &line_index, false, it))
                .collect();
            let folding_id = lsif.add(Element::Vertex(Vertex::FoldingRangeResult { result }));
            lsif.add(Element::Edge(Edge::FoldingRange(EdgeData {
                in_v: folding_id.into(),
                out_v: doc_id.into(),
            })));
            lsif.add_tokens(&line_index, doc_id, tokens);
        }
        Ok(())
    }
}
