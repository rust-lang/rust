//! LSIF (language server index format) generator

use std::env;
use std::time::Instant;

use ide::{
    Analysis, AnalysisHost, FileId, FileRange, MonikerKind, MonikerResult, PackageInformation,
    RootDatabase, StaticIndex, StaticIndexedFile, TokenId, TokenStaticData,
    VendoredLibrariesConfig,
};
use ide_db::{LineIndexDatabase, line_index::WideEncoding};
use load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace};
use lsp_types::lsif;
use project_model::{CargoConfig, ProjectManifest, ProjectWorkspace, RustLibSource};
use rustc_hash::FxHashMap;
use stdx::format_to;
use vfs::{AbsPathBuf, Vfs};

use crate::{
    cli::flags,
    line_index::{LineEndings, LineIndex, PositionEncoding},
    lsp::to_proto,
    version::version,
};

struct LsifManager<'a, 'w> {
    count: i32,
    token_map: FxHashMap<TokenId, Id>,
    range_map: FxHashMap<FileRange, Id>,
    file_map: FxHashMap<FileId, Id>,
    package_map: FxHashMap<PackageInformation, Id>,
    analysis: &'a Analysis,
    db: &'a RootDatabase,
    vfs: &'a Vfs,
    out: &'w mut dyn std::io::Write,
}

#[derive(Clone, Copy)]
struct Id(i32);

impl From<Id> for lsp_types::NumberOrString {
    fn from(Id(it): Id) -> Self {
        lsp_types::NumberOrString::Number(it)
    }
}

impl LsifManager<'_, '_> {
    fn new<'a, 'w>(
        analysis: &'a Analysis,
        db: &'a RootDatabase,
        vfs: &'a Vfs,
        out: &'w mut dyn std::io::Write,
    ) -> LsifManager<'a, 'w> {
        LsifManager {
            count: 0,
            token_map: FxHashMap::default(),
            range_map: FxHashMap::default(),
            file_map: FxHashMap::default(),
            package_map: FxHashMap::default(),
            analysis,
            db,
            vfs,
            out,
        }
    }

    fn add(&mut self, data: lsif::Element) -> Id {
        let id = Id(self.count);
        self.emit(&serde_json::to_string(&lsif::Entry { id: id.into(), data }).unwrap());
        self.count += 1;
        id
    }

    fn add_vertex(&mut self, vertex: lsif::Vertex) -> Id {
        self.add(lsif::Element::Vertex(vertex))
    }

    fn add_edge(&mut self, edge: lsif::Edge) -> Id {
        self.add(lsif::Element::Edge(edge))
    }

    fn emit(&mut self, data: &str) {
        format_to!(self.out, "{data}\n");
    }

    fn get_token_id(&mut self, id: TokenId) -> Id {
        if let Some(it) = self.token_map.get(&id) {
            return *it;
        }
        let result_set_id = self.add_vertex(lsif::Vertex::ResultSet(lsif::ResultSet { key: None }));
        self.token_map.insert(id, result_set_id);
        result_set_id
    }

    fn get_package_id(&mut self, package_information: PackageInformation) -> Id {
        if let Some(it) = self.package_map.get(&package_information) {
            return *it;
        }
        let pi = package_information.clone();
        let result_set_id =
            self.add_vertex(lsif::Vertex::PackageInformation(lsif::PackageInformation {
                name: pi.name,
                manager: "cargo".to_owned(),
                uri: None,
                content: None,
                repository: pi.repo.map(|url| lsif::Repository {
                    url,
                    r#type: "git".to_owned(),
                    commit_id: None,
                }),
                version: pi.version,
            }));
        self.package_map.insert(package_information, result_set_id);
        result_set_id
    }

    fn get_range_id(&mut self, id: FileRange) -> Id {
        if let Some(it) = self.range_map.get(&id) {
            return *it;
        }
        let file_id = id.file_id;
        let doc_id = self.get_file_id(file_id);
        let line_index = self.db.line_index(file_id);
        let line_index = LineIndex {
            index: line_index,
            encoding: PositionEncoding::Wide(WideEncoding::Utf16),
            endings: LineEndings::Unix,
        };
        let range_id = self.add_vertex(lsif::Vertex::Range {
            range: to_proto::range(&line_index, id.range),
            tag: None,
        });
        self.add_edge(lsif::Edge::Contains(lsif::EdgeDataMultiIn {
            in_vs: vec![range_id.into()],
            out_v: doc_id.into(),
        }));
        range_id
    }

    fn get_file_id(&mut self, id: FileId) -> Id {
        if let Some(it) = self.file_map.get(&id) {
            return *it;
        }
        let path = self.vfs.file_path(id);
        let path = path.as_path().unwrap();
        let doc_id = self.add_vertex(lsif::Vertex::Document(lsif::Document {
            language_id: "rust".to_owned(),
            uri: lsp_types::Url::from_file_path(path).unwrap(),
        }));
        self.file_map.insert(id, doc_id);
        doc_id
    }

    fn add_token(&mut self, id: TokenId, token: TokenStaticData) {
        let result_set_id = self.get_token_id(id);
        if let Some(hover) = token.hover {
            let hover_id = self.add_vertex(lsif::Vertex::HoverResult {
                result: lsp_types::Hover {
                    contents: lsp_types::HoverContents::Markup(to_proto::markup_content(
                        hover.markup,
                        ide::HoverDocFormat::Markdown,
                    )),
                    range: None,
                },
            });
            self.add_edge(lsif::Edge::Hover(lsif::EdgeData {
                in_v: hover_id.into(),
                out_v: result_set_id.into(),
            }));
        }
        if let Some(MonikerResult::Moniker(moniker)) = token.moniker {
            let package_id = self.get_package_id(moniker.package_information);
            let moniker_id = self.add_vertex(lsif::Vertex::Moniker(lsp_types::Moniker {
                scheme: "rust-analyzer".to_owned(),
                identifier: moniker.identifier.to_string(),
                unique: lsp_types::UniquenessLevel::Scheme,
                kind: Some(match moniker.kind {
                    MonikerKind::Import => lsp_types::MonikerKind::Import,
                    MonikerKind::Export => lsp_types::MonikerKind::Export,
                }),
            }));
            self.add_edge(lsif::Edge::PackageInformation(lsif::EdgeData {
                in_v: package_id.into(),
                out_v: moniker_id.into(),
            }));
            self.add_edge(lsif::Edge::Moniker(lsif::EdgeData {
                in_v: moniker_id.into(),
                out_v: result_set_id.into(),
            }));
        }
        if let Some(def) = token.definition {
            let result_id = self.add_vertex(lsif::Vertex::DefinitionResult);
            let def_vertex = self.get_range_id(def);
            self.add_edge(lsif::Edge::Item(lsif::Item {
                document: (*self.file_map.get(&def.file_id).unwrap()).into(),
                property: None,
                edge_data: lsif::EdgeDataMultiIn {
                    in_vs: vec![def_vertex.into()],
                    out_v: result_id.into(),
                },
            }));
            self.add_edge(lsif::Edge::Definition(lsif::EdgeData {
                in_v: result_id.into(),
                out_v: result_set_id.into(),
            }));
        }
        if !token.references.is_empty() {
            let result_id = self.add_vertex(lsif::Vertex::ReferenceResult);
            self.add_edge(lsif::Edge::References(lsif::EdgeData {
                in_v: result_id.into(),
                out_v: result_set_id.into(),
            }));
            let mut edges = token.references.iter().fold(
                FxHashMap::<_, Vec<lsp_types::NumberOrString>>::default(),
                |mut edges, it| {
                    let entry = edges.entry((it.range.file_id, it.is_definition)).or_default();
                    entry.push((*self.range_map.get(&it.range).unwrap()).into());
                    edges
                },
            );
            for it in token.references {
                if let Some(vertices) = edges.remove(&(it.range.file_id, it.is_definition)) {
                    self.add_edge(lsif::Edge::Item(lsif::Item {
                        document: (*self.file_map.get(&it.range.file_id).unwrap()).into(),
                        property: Some(if it.is_definition {
                            lsif::ItemKind::Definitions
                        } else {
                            lsif::ItemKind::References
                        }),
                        edge_data: lsif::EdgeDataMultiIn {
                            in_vs: vertices,
                            out_v: result_id.into(),
                        },
                    }));
                }
            }
        }
    }

    fn add_file(&mut self, file: StaticIndexedFile) {
        let StaticIndexedFile { file_id, tokens, folds, .. } = file;
        let doc_id = self.get_file_id(file_id);
        let text = self.analysis.file_text(file_id).unwrap();
        let line_index = self.db.line_index(file_id);
        let line_index = LineIndex {
            index: line_index,
            encoding: PositionEncoding::Wide(WideEncoding::Utf16),
            endings: LineEndings::Unix,
        };
        let result = folds
            .into_iter()
            .map(|it| to_proto::folding_range(&text, &line_index, false, it))
            .collect();
        let folding_id = self.add_vertex(lsif::Vertex::FoldingRangeResult { result });
        self.add_edge(lsif::Edge::FoldingRange(lsif::EdgeData {
            in_v: folding_id.into(),
            out_v: doc_id.into(),
        }));
        let tokens_id = tokens
            .into_iter()
            .map(|(range, id)| {
                let range_id = self.add_vertex(lsif::Vertex::Range {
                    range: to_proto::range(&line_index, range),
                    tag: None,
                });
                self.range_map.insert(FileRange { file_id, range }, range_id);
                let result_set_id = self.get_token_id(id);
                self.add_edge(lsif::Edge::Next(lsif::EdgeData {
                    in_v: result_set_id.into(),
                    out_v: range_id.into(),
                }));
                range_id.into()
            })
            .collect();
        self.add_edge(lsif::Edge::Contains(lsif::EdgeDataMultiIn {
            in_vs: tokens_id,
            out_v: doc_id.into(),
        }));
    }
}

impl flags::Lsif {
    pub fn run(
        self,
        out: &mut dyn std::io::Write,
        sysroot: Option<RustLibSource>,
    ) -> anyhow::Result<()> {
        let now = Instant::now();
        let cargo_config =
            &CargoConfig { sysroot, all_targets: true, set_test: true, ..Default::default() };
        let no_progress = &|_| ();
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: true,
            with_proc_macro_server: ProcMacroServerChoice::Sysroot,
            prefill_caches: false,
        };
        let path = AbsPathBuf::assert_utf8(env::current_dir()?.join(self.path));
        let root = ProjectManifest::discover_single(&path)?;
        eprintln!("Generating LSIF for project at {root}");
        let mut workspace = ProjectWorkspace::load(root, cargo_config, no_progress)?;

        let build_scripts = workspace.run_build_scripts(cargo_config, no_progress)?;
        workspace.set_build_scripts(build_scripts);

        let (db, vfs, _proc_macro) =
            load_workspace(workspace, &cargo_config.extra_env, &load_cargo_config)?;
        let host = AnalysisHost::with_database(db);
        let db = host.raw_database();
        let analysis = host.analysis();

        let vendored_libs_config = if self.exclude_vendored_libraries {
            VendoredLibrariesConfig::Excluded
        } else {
            VendoredLibrariesConfig::Included { workspace_root: &path.clone().into() }
        };

        let si = StaticIndex::compute(&analysis, vendored_libs_config);

        let mut lsif = LsifManager::new(&analysis, db, &vfs, out);
        lsif.add_vertex(lsif::Vertex::MetaData(lsif::MetaData {
            version: String::from("0.5.0"),
            project_root: lsp_types::Url::from_file_path(path).unwrap(),
            position_encoding: lsif::Encoding::Utf16,
            tool_info: Some(lsp_types::lsif::ToolInfo {
                name: "rust-analyzer".to_owned(),
                args: vec![],
                version: Some(version().to_string()),
            }),
        }));
        for file in si.files {
            lsif.add_file(file);
        }
        for (id, token) in si.tokens.iter() {
            lsif.add_token(id, token);
        }
        eprintln!("Generating LSIF finished in {:?}", now.elapsed());
        Ok(())
    }
}
