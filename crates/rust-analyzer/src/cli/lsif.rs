//! Lsif generator

use std::collections::HashMap;
use std::env;
use std::time::Instant;

use ide::{Analysis, Cancellable, RootDatabase, StaticIndex, StaticIndexedFile, TokenId, TokenStaticData};
use ide_db::LineIndexDatabase;

use ide_db::base_db::salsa::{self, ParallelDatabase};
use lsp_types::{lsif::*, Hover, HoverContents, NumberOrString};
use project_model::{CargoConfig, ProjectManifest, ProjectWorkspace};
use vfs::{AbsPathBuf, Vfs};

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

struct LsifManager<'a> {
    count: i32,
    token_map: HashMap<TokenId, Id>,
    analysis: &'a Analysis,
    db: &'a RootDatabase,
    vfs: &'a Vfs,
}

#[derive(Clone, Copy)]
struct Id(i32);

impl From<Id> for NumberOrString {
    fn from(Id(x): Id) -> Self {
        NumberOrString::Number(x)
    }
}

impl LsifManager<'_> {
    fn new<'a>(analysis: &'a Analysis, db: &'a RootDatabase, vfs: &'a Vfs) -> LsifManager<'a> {
        LsifManager {
            count: 0,
            token_map: HashMap::default(),
            analysis,
            db,
            vfs,
        }
    }
    
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

    fn add_token(&mut self, id: TokenId, token: TokenStaticData) {
        let result_set_id = self.add(Element::Vertex(Vertex::ResultSet(ResultSet { key: None })));
        self.token_map.insert(id, result_set_id);
        if let Some(hover) = token.hover {
            let hover_id = self.add(Element::Vertex(Vertex::HoverResult {
                result: Hover {
                    contents: HoverContents::Markup(to_proto::markup_content(hover.markup)),
                    range: None,
                },
            }));
            self.add(Element::Edge(Edge::Hover(EdgeData {
                in_v: hover_id.into(),
                out_v: result_set_id.into(),
            })));
        }
    }

    fn add_file(&mut self, file: StaticIndexedFile) -> Cancellable<()> {
        let StaticIndexedFile { file_id, tokens, folds} = file;
        let path = self.vfs.file_path(file_id);
        let path = path.as_path().unwrap();
        let doc_id = self.add(Element::Vertex(Vertex::Document(Document {
            language_id: "rust".to_string(),
            uri: lsp_types::Url::from_file_path(path).unwrap(),
        })));
        let text = self.analysis.file_text(file_id)?;
        let line_index = self.db.line_index(file_id);
        let line_index = LineIndex {
            index: line_index.clone(),
            encoding: OffsetEncoding::Utf16,
            endings: LineEndings::Unix,
        };
        let result = folds
            .into_iter()
            .map(|it| to_proto::folding_range(&*text, &line_index, false, it))
            .collect();
        let folding_id = self.add(Element::Vertex(Vertex::FoldingRangeResult { result }));
        self.add(Element::Edge(Edge::FoldingRange(EdgeData {
            in_v: folding_id.into(),
            out_v: doc_id.into(),
        })));
        let tokens_id = tokens
            .into_iter()
            .map(|(range, id)| {
                let range_id = self.add(Element::Vertex(Vertex::Range {
                    range: to_proto::range(&line_index, range),
                    tag: None,
                }));
                let result_set_id = *self.token_map.get(&id).expect("token map doesn't contain id");
                self.add(Element::Edge(Edge::Next(EdgeData {
                    in_v: result_set_id.into(),
                    out_v: range_id.into(),
                })));
                range_id.into()
            })
            .collect();
        self.add(Element::Edge(Edge::Contains(EdgeDataMultiIn {
            in_vs: tokens_id,
            out_v: doc_id.into(),
        })));
        Ok(())
    }
}

impl flags::Lsif {
    pub fn run(self) -> Result<()> {
        eprintln!("Generating LSIF started...");
        let now = Instant::now();
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

        let mut lsif = LsifManager::new(&analysis, db, &vfs);
        lsif.add(Element::Vertex(Vertex::MetaData(MetaData {
            version: String::from("0.5.0"),
            project_root: lsp_types::Url::from_file_path(path).unwrap(),
            position_encoding: Encoding::Utf16,
            tool_info: None,
        })));
        for (id, token) in si.tokens.iter() {
            lsif.add_token(id, token);
        }
        for file in si.files {
            lsif.add_file(file)?;
        }
        eprintln!("Generating LSIF finished in {:?}", now.elapsed());
        Ok(())
    }
}
