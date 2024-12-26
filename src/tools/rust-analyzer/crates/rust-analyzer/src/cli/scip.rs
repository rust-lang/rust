//! SCIP generator

use std::{path::PathBuf, time::Instant};

use ide::{
    AnalysisHost, LineCol, Moniker, MonikerDescriptorKind, MonikerIdentifier, MonikerResult,
    StaticIndex, StaticIndexedFile, SymbolInformationKind, TextRange, TokenId, TokenStaticData,
    VendoredLibrariesConfig,
};
use ide_db::LineIndexDatabase;
use load_cargo::{load_workspace_at, LoadCargoConfig, ProcMacroServerChoice};
use rustc_hash::{FxHashMap, FxHashSet};
use scip::types as scip_types;
use tracing::error;

use crate::{
    cli::flags,
    config::ConfigChange,
    line_index::{LineEndings, LineIndex, PositionEncoding},
};

impl flags::Scip {
    pub fn run(self) -> anyhow::Result<()> {
        eprintln!("Generating SCIP start...");
        let now = Instant::now();

        let no_progress = &|s| (eprintln!("rust-analyzer: Loading {s}"));
        let root =
            vfs::AbsPathBuf::assert_utf8(std::env::current_dir()?.join(&self.path)).normalize();

        let mut config = crate::config::Config::new(
            root.clone(),
            lsp_types::ClientCapabilities::default(),
            vec![],
            None,
        );

        if let Some(p) = self.config_path {
            let mut file = std::io::BufReader::new(std::fs::File::open(p)?);
            let json = serde_json::from_reader(&mut file)?;
            let mut change = ConfigChange::default();
            change.change_client_config(json);

            let error_sink;
            (config, error_sink, _) = config.apply_change(change);

            // FIXME @alibektas : What happens to errors without logging?
            error!(?error_sink, "Config Error(s)");
        }
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: true,
            with_proc_macro_server: ProcMacroServerChoice::Sysroot,
            prefill_caches: true,
        };
        let cargo_config = config.cargo(None);
        let (db, vfs, _) = load_workspace_at(
            root.as_path().as_ref(),
            &cargo_config,
            &load_cargo_config,
            &no_progress,
        )?;
        let host = AnalysisHost::with_database(db);
        let db = host.raw_database();
        let analysis = host.analysis();

        let vendored_libs_config = if self.exclude_vendored_libraries {
            VendoredLibrariesConfig::Excluded
        } else {
            VendoredLibrariesConfig::Included { workspace_root: &root.clone().into() }
        };

        let si = StaticIndex::compute(&analysis, vendored_libs_config);

        let metadata = scip_types::Metadata {
            version: scip_types::ProtocolVersion::UnspecifiedProtocolVersion.into(),
            tool_info: Some(scip_types::ToolInfo {
                name: "rust-analyzer".to_owned(),
                version: format!("{}", crate::version::version()),
                arguments: vec![],
                special_fields: Default::default(),
            })
            .into(),
            project_root: format!("file://{root}"),
            text_document_encoding: scip_types::TextEncoding::UTF8.into(),
            special_fields: Default::default(),
        };
        let mut documents = Vec::new();

        let mut token_ids_emitted: FxHashSet<TokenId> = FxHashSet::default();
        let mut global_symbols_emitted: FxHashSet<String> = FxHashSet::default();
        let mut duplicate_symbols: Vec<(String, String)> = Vec::new();
        let mut symbol_generator = SymbolGenerator::new();

        for StaticIndexedFile { file_id, tokens, .. } in si.files {
            symbol_generator.clear_document_local_state();

            let relative_path = match get_relative_filepath(&vfs, &root, file_id) {
                Some(relative_path) => relative_path,
                None => continue,
            };

            let line_index = LineIndex {
                index: db.line_index(file_id),
                encoding: PositionEncoding::Utf8,
                endings: LineEndings::Unix,
            };

            let mut occurrences = Vec::new();
            let mut symbols = Vec::new();

            tokens.into_iter().for_each(|(text_range, id)| {
                let token = si.tokens.get(id).unwrap();

                let (symbol, enclosing_symbol) =
                    if let Some(TokenSymbols { symbol, enclosing_symbol }) =
                        symbol_generator.token_symbols(id, token)
                    {
                        (symbol, enclosing_symbol)
                    } else {
                        ("".to_owned(), None)
                    };

                if !symbol.is_empty() && token_ids_emitted.insert(id) {
                    if !symbol.starts_with("local ")
                        && !global_symbols_emitted.insert(symbol.clone())
                    {
                        let source_location =
                            text_range_to_string(relative_path.as_str(), &line_index, text_range);
                        duplicate_symbols.push((source_location, symbol.clone()));
                    } else {
                        let documentation = match &token.documentation {
                            Some(doc) => vec![doc.as_str().to_owned()],
                            None => vec![],
                        };

                        let position_encoding =
                            scip_types::PositionEncoding::UTF8CodeUnitOffsetFromLineStart.into();
                        let signature_documentation =
                            token.signature.clone().map(|text| scip_types::Document {
                                relative_path: relative_path.clone(),
                                language: "rust".to_owned(),
                                text,
                                position_encoding,
                                ..Default::default()
                            });
                        let symbol_info = scip_types::SymbolInformation {
                            symbol: symbol.clone(),
                            documentation,
                            relationships: Vec::new(),
                            special_fields: Default::default(),
                            kind: symbol_kind(token.kind).into(),
                            display_name: token.display_name.clone().unwrap_or_default(),
                            signature_documentation: signature_documentation.into(),
                            enclosing_symbol: enclosing_symbol.unwrap_or_default(),
                        };

                        symbols.push(symbol_info)
                    }
                }

                // If the range of the def and the range of the token are the same, this must be the definition.
                // they also must be in the same file. See https://github.com/rust-lang/rust-analyzer/pull/17988
                let mut symbol_roles = Default::default();
                match token.definition {
                    Some(def) if def.file_id == file_id && def.range == text_range => {
                        symbol_roles |= scip_types::SymbolRole::Definition as i32;
                    }
                    _ => {}
                };

                occurrences.push(scip_types::Occurrence {
                    range: text_range_to_scip_range(&line_index, text_range),
                    symbol,
                    symbol_roles,
                    override_documentation: Vec::new(),
                    syntax_kind: Default::default(),
                    diagnostics: Vec::new(),
                    special_fields: Default::default(),
                    enclosing_range: Vec::new(),
                });
            });

            if occurrences.is_empty() {
                continue;
            }

            let position_encoding =
                scip_types::PositionEncoding::UTF8CodeUnitOffsetFromLineStart.into();
            documents.push(scip_types::Document {
                relative_path,
                language: "rust".to_owned(),
                occurrences,
                symbols,
                text: String::new(),
                position_encoding,
                special_fields: Default::default(),
            });
        }

        let index = scip_types::Index {
            metadata: Some(metadata).into(),
            documents,
            external_symbols: Vec::new(),
            special_fields: Default::default(),
        };

        if !duplicate_symbols.is_empty() {
            eprintln!("{}", DUPLICATE_SYMBOLS_MESSAGE);
            for (source_location, symbol) in duplicate_symbols {
                eprintln!("{}", source_location);
                eprintln!("  Duplicate symbol: {}", symbol);
                eprintln!();
            }
        }

        let out_path = self.output.unwrap_or_else(|| PathBuf::from(r"index.scip"));
        scip::write_message_to_file(out_path, index)
            .map_err(|err| anyhow::format_err!("Failed to write scip to file: {}", err))?;

        eprintln!("Generating SCIP finished {:?}", now.elapsed());
        Ok(())
    }
}

// TODO: Fix the known buggy cases described here.
const DUPLICATE_SYMBOLS_MESSAGE: &str = "
Encountered duplicate scip symbols, indicating an internal rust-analyzer bug. These duplicates are
included in the output, but this causes information lookup to be ambiguous and so information about
these symbols presented by downstream tools may be incorrect.

Known cases that can cause this:

  * Definitions in crate example binaries which have the same symbol as definitions in the library
  or some other example.

  * When a struct/enum/const/static/impl is defined with a function, it erroneously appears to be
  defined at the same level as the function.

Duplicate symbols encountered:
";

fn get_relative_filepath(
    vfs: &vfs::Vfs,
    rootpath: &vfs::AbsPathBuf,
    file_id: ide::FileId,
) -> Option<String> {
    Some(vfs.file_path(file_id).as_path()?.strip_prefix(rootpath)?.as_str().to_owned())
}

// SCIP Ranges have a (very large) optimization that ranges if they are on the same line
// only encode as a vector of [start_line, start_col, end_col].
//
// This transforms a line index into the optimized SCIP Range.
fn text_range_to_scip_range(line_index: &LineIndex, range: TextRange) -> Vec<i32> {
    let LineCol { line: start_line, col: start_col } = line_index.index.line_col(range.start());
    let LineCol { line: end_line, col: end_col } = line_index.index.line_col(range.end());

    if start_line == end_line {
        vec![start_line as i32, start_col as i32, end_col as i32]
    } else {
        vec![start_line as i32, start_col as i32, end_line as i32, end_col as i32]
    }
}

fn text_range_to_string(relative_path: &str, line_index: &LineIndex, range: TextRange) -> String {
    let LineCol { line: start_line, col: start_col } = line_index.index.line_col(range.start());
    let LineCol { line: end_line, col: end_col } = line_index.index.line_col(range.end());

    format!("{relative_path}:{start_line}:{start_col}-{end_line}:{end_col}")
}

fn new_descriptor_str(
    name: &str,
    suffix: scip_types::descriptor::Suffix,
) -> scip_types::Descriptor {
    scip_types::Descriptor {
        name: name.to_owned(),
        disambiguator: "".to_owned(),
        suffix: suffix.into(),
        special_fields: Default::default(),
    }
}

fn symbol_kind(kind: SymbolInformationKind) -> scip_types::symbol_information::Kind {
    use scip_types::symbol_information::Kind as ScipKind;
    match kind {
        SymbolInformationKind::AssociatedType => ScipKind::AssociatedType,
        SymbolInformationKind::Attribute => ScipKind::Attribute,
        SymbolInformationKind::Constant => ScipKind::Constant,
        SymbolInformationKind::Enum => ScipKind::Enum,
        SymbolInformationKind::EnumMember => ScipKind::EnumMember,
        SymbolInformationKind::Field => ScipKind::Field,
        SymbolInformationKind::Function => ScipKind::Function,
        SymbolInformationKind::Macro => ScipKind::Macro,
        SymbolInformationKind::Method => ScipKind::Method,
        SymbolInformationKind::Module => ScipKind::Module,
        SymbolInformationKind::Parameter => ScipKind::Parameter,
        SymbolInformationKind::SelfParameter => ScipKind::SelfParameter,
        SymbolInformationKind::StaticMethod => ScipKind::StaticMethod,
        SymbolInformationKind::StaticVariable => ScipKind::StaticVariable,
        SymbolInformationKind::Struct => ScipKind::Struct,
        SymbolInformationKind::Trait => ScipKind::Trait,
        SymbolInformationKind::TraitMethod => ScipKind::TraitMethod,
        SymbolInformationKind::Type => ScipKind::Type,
        SymbolInformationKind::TypeAlias => ScipKind::TypeAlias,
        SymbolInformationKind::TypeParameter => ScipKind::TypeParameter,
        SymbolInformationKind::Union => ScipKind::Union,
        SymbolInformationKind::Variable => ScipKind::Variable,
    }
}

#[derive(Clone)]
struct TokenSymbols {
    symbol: String,
    /// Definition that contains this one. Only set when `symbol` is local.
    enclosing_symbol: Option<String>,
}

struct SymbolGenerator {
    token_to_symbols: FxHashMap<TokenId, Option<TokenSymbols>>,
    local_count: usize,
}

impl SymbolGenerator {
    fn new() -> Self {
        SymbolGenerator { token_to_symbols: FxHashMap::default(), local_count: 0 }
    }

    fn clear_document_local_state(&mut self) {
        self.local_count = 0;
    }

    fn token_symbols(&mut self, id: TokenId, token: &TokenStaticData) -> Option<TokenSymbols> {
        let mut local_count = self.local_count;
        let token_symbols = self
            .token_to_symbols
            .entry(id)
            .or_insert_with(|| {
                Some(match token.moniker.as_ref()? {
                    MonikerResult::Moniker(moniker) => TokenSymbols {
                        symbol: scip::symbol::format_symbol(moniker_to_symbol(moniker)),
                        enclosing_symbol: None,
                    },
                    MonikerResult::Local { enclosing_moniker } => {
                        let local_symbol = scip::types::Symbol::new_local(local_count);
                        local_count += 1;
                        TokenSymbols {
                            symbol: scip::symbol::format_symbol(local_symbol),
                            enclosing_symbol: enclosing_moniker
                                .as_ref()
                                .map(moniker_to_symbol)
                                .map(scip::symbol::format_symbol),
                        }
                    }
                })
            })
            .clone();
        self.local_count = local_count;
        token_symbols
    }
}

fn moniker_to_symbol(moniker: &Moniker) -> scip_types::Symbol {
    scip_types::Symbol {
        scheme: "rust-analyzer".into(),
        package: Some(scip_types::Package {
            manager: "cargo".to_owned(),
            name: moniker.package_information.name.clone(),
            version: moniker.package_information.version.clone().unwrap_or_else(|| ".".to_owned()),
            special_fields: Default::default(),
        })
        .into(),
        descriptors: moniker_descriptors(&moniker.identifier),
        special_fields: Default::default(),
    }
}

fn moniker_descriptors(identifier: &MonikerIdentifier) -> Vec<scip_types::Descriptor> {
    use scip_types::descriptor::Suffix::*;
    identifier
        .description
        .iter()
        .map(|desc| {
            new_descriptor_str(
                &desc.name,
                match desc.desc {
                    MonikerDescriptorKind::Namespace => Namespace,
                    MonikerDescriptorKind::Type => Type,
                    MonikerDescriptorKind::Term => Term,
                    MonikerDescriptorKind::Method => Method,
                    MonikerDescriptorKind::TypeParameter => TypeParameter,
                    MonikerDescriptorKind::Parameter => Parameter,
                    MonikerDescriptorKind::Macro => Macro,
                    MonikerDescriptorKind::Meta => Meta,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use ide::{FilePosition, TextSize};
    use test_fixture::ChangeFixture;
    use vfs::VfsPath;

    fn position(ra_fixture: &str) -> (AnalysisHost, FilePosition) {
        let mut host = AnalysisHost::default();
        let change_fixture = ChangeFixture::parse(ra_fixture);
        host.raw_database_mut().apply_change(change_fixture.change);
        let (file_id, range_or_offset) =
            change_fixture.file_position.expect("expected a marker ()");
        let offset = range_or_offset.expect_offset();
        (host, FilePosition { file_id: file_id.into(), offset })
    }

    /// If expected == "", then assert that there are no symbols (this is basically local symbol)
    #[track_caller]
    fn check_symbol(ra_fixture: &str, expected: &str) {
        let (host, position) = position(ra_fixture);

        let analysis = host.analysis();
        let si = StaticIndex::compute(
            &analysis,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );

        let FilePosition { file_id, offset } = position;

        let mut found_symbol = None;
        for file in &si.files {
            if file.file_id != file_id {
                continue;
            }
            for &(range, id) in &file.tokens {
                if range.contains(offset - TextSize::from(1)) {
                    let token = si.tokens.get(id).unwrap();
                    found_symbol = match token.moniker.as_ref() {
                        None => None,
                        Some(MonikerResult::Moniker(moniker)) => {
                            Some(scip::symbol::format_symbol(moniker_to_symbol(moniker)))
                        }
                        Some(MonikerResult::Local { enclosing_moniker: Some(moniker) }) => {
                            Some(format!(
                                "local enclosed by {}",
                                scip::symbol::format_symbol(moniker_to_symbol(moniker))
                            ))
                        }
                        Some(MonikerResult::Local { enclosing_moniker: None }) => {
                            Some("unenclosed local".to_owned())
                        }
                    };
                    break;
                }
            }
        }

        if expected.is_empty() {
            assert!(found_symbol.is_none(), "must have no symbols {found_symbol:?}");
            return;
        }

        assert!(found_symbol.is_some(), "must have one symbol {found_symbol:?}");
        assert_eq!(found_symbol.unwrap(), expected);
    }

    #[test]
    fn basic() {
        check_symbol(
            r#"
//- /workspace/lib.rs crate:main deps:foo
use foo::example_mod::func;
fn main() {
    func$0();
}
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod example_mod {
    pub fn func() {}
}
"#,
            "rust-analyzer cargo foo 0.1.0 example_mod/func().",
        );
    }

    #[test]
    fn symbol_for_trait() {
        check_symbol(
            r#"
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod module {
    pub trait MyTrait {
        pub fn func$0() {}
    }
}
"#,
            "rust-analyzer cargo foo 0.1.0 module/MyTrait#func().",
        );
    }

    #[test]
    fn symbol_for_trait_alias() {
        check_symbol(
            r#"
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
#![feature(trait_alias)]
pub mod module {
    pub trait MyTrait {}
    pub trait MyTraitAlias$0 = MyTrait;
}
"#,
            "rust-analyzer cargo foo 0.1.0 module/MyTraitAlias#",
        );
    }

    #[test]
    fn symbol_for_trait_constant() {
        check_symbol(
            r#"
    //- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
    pub mod module {
        pub trait MyTrait {
            const MY_CONST$0: u8;
        }
    }
    "#,
            "rust-analyzer cargo foo 0.1.0 module/MyTrait#MY_CONST.",
        );
    }

    #[test]
    fn symbol_for_trait_type() {
        check_symbol(
            r#"
    //- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
    pub mod module {
        pub trait MyTrait {
            type MyType$0;
        }
    }
    "#,
            "rust-analyzer cargo foo 0.1.0 module/MyTrait#MyType#",
        );
    }

    #[test]
    fn symbol_for_trait_impl_function() {
        check_symbol(
            r#"
    //- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
    pub mod module {
        pub trait MyTrait {
            pub fn func() {}
        }

        struct MyStruct {}

        impl MyTrait for MyStruct {
            pub fn func$0() {}
        }
    }
    "#,
            "rust-analyzer cargo foo 0.1.0 module/impl#[MyStruct][MyTrait]func().",
        );
    }

    #[test]
    fn symbol_for_field() {
        check_symbol(
            r#"
    //- /workspace/lib.rs crate:main deps:foo
    use foo::St;
    fn main() {
        let x = St { a$0: 2 };
    }
    //- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
    pub struct St {
        pub a: i32,
    }
    "#,
            "rust-analyzer cargo foo 0.1.0 St#a.",
        );
    }

    #[test]
    fn symbol_for_param() {
        check_symbol(
            r#"
//- /workspace/lib.rs crate:main deps:foo
use foo::example_mod::func;
fn main() {
    func(42);
}
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod example_mod {
    pub fn func(x$0: usize) {}
}
"#,
            "local enclosed by rust-analyzer cargo foo 0.1.0 example_mod/func().",
        );
    }

    #[test]
    fn symbol_for_closure_param() {
        check_symbol(
            r#"
//- /workspace/lib.rs crate:main deps:foo
use foo::example_mod::func;
fn main() {
    func();
}
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod example_mod {
    pub fn func() {
        let f = |x$0: usize| {};
    }
}
"#,
            "local enclosed by rust-analyzer cargo foo 0.1.0 example_mod/func().",
        );
    }

    #[test]
    fn local_symbol_for_local() {
        check_symbol(
            r#"
    //- /workspace/lib.rs crate:main deps:foo
    use foo::module::func;
    fn main() {
        func();
    }
    //- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
    pub mod module {
        pub fn func() {
            let x$0 = 2;
        }
    }
    "#,
            "local enclosed by rust-analyzer cargo foo 0.1.0 module/func().",
        );
    }

    #[test]
    fn global_symbol_for_pub_struct() {
        check_symbol(
            r#"
    //- /workspace/lib.rs crate:main
    mod foo;

    fn main() {
        let _bar = foo::Bar { i: 0 };
    }
    //- /workspace/foo.rs
    pub struct Bar$0 {
        pub i: i32,
    }
    "#,
            "rust-analyzer cargo main . foo/Bar#",
        );
    }

    #[test]
    fn global_symbol_for_pub_struct_reference() {
        check_symbol(
            r#"
    //- /workspace/lib.rs crate:main
    mod foo;

    fn main() {
        let _bar = foo::Bar$0 { i: 0 };
    }
    //- /workspace/foo.rs
    pub struct Bar {
        pub i: i32,
    }
    "#,
            "rust-analyzer cargo main . foo/Bar#",
        );
    }

    #[test]
    fn symbol_for_type_alias() {
        check_symbol(
            r#"
    //- /workspace/lib.rs crate:main
    pub type MyTypeAlias$0 = u8;
    "#,
            "rust-analyzer cargo main . MyTypeAlias#",
        );
    }

    // TODO: This test represents current misbehavior.
    #[test]
    fn symbol_for_nested_function() {
        check_symbol(
            r#"
    //- /workspace/lib.rs crate:main
    pub fn func() {
       pub fn inner_func$0() {}
    }
    "#,
            "rust-analyzer cargo main . inner_func().",
            // TODO: This should be a local:
            // "local enclosed by rust-analyzer cargo main . func().",
        );
    }

    // TODO: This test represents current misbehavior.
    #[test]
    fn symbol_for_struct_in_function() {
        check_symbol(
            r#"
    //- /workspace/lib.rs crate:main
    pub fn func() {
       struct SomeStruct$0 {}
    }
    "#,
            "rust-analyzer cargo main . SomeStruct#",
            // TODO: This should be a local:
            // "local enclosed by rust-analyzer cargo main . func().",
        );
    }

    // TODO: This test represents current misbehavior.
    #[test]
    fn symbol_for_const_in_function() {
        check_symbol(
            r#"
    //- /workspace/lib.rs crate:main
    pub fn func() {
       const SOME_CONST$0: u32 = 1;
    }
    "#,
            "rust-analyzer cargo main . SOME_CONST.",
            // TODO: This should be a local:
            // "local enclosed by rust-analyzer cargo main . func().",
        );
    }

    // TODO: This test represents current misbehavior.
    #[test]
    fn symbol_for_static_in_function() {
        check_symbol(
            r#"
    //- /workspace/lib.rs crate:main
    pub fn func() {
       static SOME_STATIC$0: u32 = 1;
    }
    "#,
            "rust-analyzer cargo main . SOME_STATIC.",
            // TODO: This should be a local:
            // "local enclosed by rust-analyzer cargo main . func().",
        );
    }

    #[test]
    fn documentation_matches_doc_comment() {
        let s = "/// foo\nfn bar() {}";

        let mut host = AnalysisHost::default();
        let change_fixture = ChangeFixture::parse(s);
        host.raw_database_mut().apply_change(change_fixture.change);

        let analysis = host.analysis();
        let si = StaticIndex::compute(
            &analysis,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );

        let file = si.files.first().unwrap();
        let (_, token_id) = file.tokens.first().unwrap();
        let token = si.tokens.get(*token_id).unwrap();

        assert_eq!(token.documentation.as_ref().map(|d| d.as_str()), Some("foo"));
    }
}
