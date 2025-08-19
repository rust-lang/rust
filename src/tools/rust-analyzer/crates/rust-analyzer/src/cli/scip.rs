//! SCIP generator

use std::{path::PathBuf, time::Instant};

use ide::{
    AnalysisHost, LineCol, Moniker, MonikerDescriptorKind, MonikerIdentifier, MonikerResult,
    RootDatabase, StaticIndex, StaticIndexedFile, SymbolInformationKind, TextRange, TokenId,
    TokenStaticData, VendoredLibrariesConfig,
};
use ide_db::LineIndexDatabase;
use load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace_at};
use rustc_hash::{FxHashMap, FxHashSet};
use scip::types::{self as scip_types, SymbolInformation};
use tracing::error;
use vfs::FileId;

use crate::{
    cli::flags,
    config::ConfigChange,
    line_index::{LineEndings, LineIndex, PositionEncoding},
};

impl flags::Scip {
    pub fn run(self) -> anyhow::Result<()> {
        eprintln!("Generating SCIP start...");
        let now = Instant::now();

        let no_progress = &|s| eprintln!("rust-analyzer: Loading {s}");
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

        // All TokenIds where an Occurrence has been emitted that references a symbol.
        let mut token_ids_referenced: FxHashSet<TokenId> = FxHashSet::default();
        // All TokenIds where the SymbolInformation has been written to the document.
        let mut token_ids_emitted: FxHashSet<TokenId> = FxHashSet::default();
        // All FileIds emitted as documents.
        let mut file_ids_emitted: FxHashSet<FileId> = FxHashSet::default();

        // All non-local symbols encountered, for detecting duplicate symbol errors.
        let mut nonlocal_symbols_emitted: FxHashSet<String> = FxHashSet::default();
        // List of (source_location, symbol) for duplicate symbol errors to report.
        let mut duplicate_symbol_errors: Vec<(String, String)> = Vec::new();
        // This is called after definitions have been deduplicated by token_ids_emitted. The purpose
        // is to detect reuse of symbol names because this causes ambiguity about their meaning.
        let mut record_error_if_symbol_already_used =
            |symbol: String,
             is_inherent_impl: bool,
             relative_path: &str,
             line_index: &LineIndex,
             text_range: TextRange| {
                let is_local = symbol.starts_with("local ");
                if !is_local && !nonlocal_symbols_emitted.insert(symbol.clone()) {
                    if is_inherent_impl {
                        // FIXME: See #18772. Duplicate SymbolInformation for inherent impls is
                        // omitted. It would be preferable to emit them with numbers with
                        // disambiguation, but this is more complex to implement.
                        false
                    } else {
                        let source_location =
                            text_range_to_string(relative_path, line_index, text_range);
                        duplicate_symbol_errors.push((source_location, symbol));
                        // Keep duplicate SymbolInformation. This behavior is preferred over
                        // omitting so that the issue might be visible within downstream tools.
                        true
                    }
                } else {
                    true
                }
            };

        // Generates symbols from token monikers.
        let mut symbol_generator = SymbolGenerator::default();

        for StaticIndexedFile { file_id, tokens, .. } in si.files {
            symbol_generator.clear_document_local_state();

            let Some(relative_path) = get_relative_filepath(&vfs, &root, file_id) else { continue };
            let line_index = get_line_index(db, file_id);

            let mut occurrences = Vec::new();
            let mut symbols = Vec::new();

            for (text_range, id) in tokens.into_iter() {
                let token = si.tokens.get(id).unwrap();

                let Some(TokenSymbols { symbol, enclosing_symbol, is_inherent_impl }) =
                    symbol_generator.token_symbols(id, token)
                else {
                    // token did not have a moniker, so there is no reasonable occurrence to emit
                    // see ide::moniker::def_to_moniker
                    continue;
                };

                let is_defined_in_this_document = match token.definition {
                    Some(def) => def.file_id == file_id,
                    _ => false,
                };
                if is_defined_in_this_document {
                    if token_ids_emitted.insert(id) {
                        // token_ids_emitted does deduplication. This checks that this results
                        // in unique emitted symbols, as otherwise references are ambiguous.
                        let should_emit = record_error_if_symbol_already_used(
                            symbol.clone(),
                            is_inherent_impl,
                            relative_path.as_str(),
                            &line_index,
                            text_range,
                        );
                        if should_emit {
                            symbols.push(compute_symbol_info(
                                symbol.clone(),
                                enclosing_symbol,
                                token,
                            ));
                        }
                    }
                } else {
                    token_ids_referenced.insert(id);
                }

                // If the range of the def and the range of the token are the same, this must be the definition.
                // they also must be in the same file. See https://github.com/rust-lang/rust-analyzer/pull/17988
                let is_definition = match token.definition {
                    Some(def) => def.file_id == file_id && def.range == text_range,
                    _ => false,
                };

                let mut symbol_roles = Default::default();
                if is_definition {
                    symbol_roles |= scip_types::SymbolRole::Definition as i32;
                }

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
            }

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
            if !file_ids_emitted.insert(file_id) {
                panic!("Invariant violation: file emitted multiple times.");
            }
        }

        // Collect all symbols referenced by the files but not defined within them.
        let mut external_symbols = Vec::new();
        for id in token_ids_referenced.difference(&token_ids_emitted) {
            let id = *id;
            let token = si.tokens.get(id).unwrap();

            let Some(definition) = token.definition else {
                break;
            };

            let file_id = definition.file_id;
            let Some(relative_path) = get_relative_filepath(&vfs, &root, file_id) else { continue };
            let line_index = get_line_index(db, file_id);
            let text_range = definition.range;
            if file_ids_emitted.contains(&file_id) {
                tracing::error!(
                    "Bug: definition at {} should have been in an SCIP document but was not.",
                    text_range_to_string(relative_path.as_str(), &line_index, text_range)
                );
                continue;
            }

            let TokenSymbols { symbol, enclosing_symbol, .. } = symbol_generator
                .token_symbols(id, token)
                .expect("To have been referenced, the symbol must be in the cache.");

            record_error_if_symbol_already_used(
                symbol.clone(),
                false,
                relative_path.as_str(),
                &line_index,
                text_range,
            );
            external_symbols.push(compute_symbol_info(symbol.clone(), enclosing_symbol, token));
        }

        let index = scip_types::Index {
            metadata: Some(metadata).into(),
            documents,
            external_symbols,
            special_fields: Default::default(),
        };

        if !duplicate_symbol_errors.is_empty() {
            eprintln!("{DUPLICATE_SYMBOLS_MESSAGE}");
            for (source_location, symbol) in duplicate_symbol_errors {
                eprintln!("{source_location}");
                eprintln!("  Duplicate symbol: {symbol}");
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

// FIXME: Known buggy cases are described here.
const DUPLICATE_SYMBOLS_MESSAGE: &str = "
Encountered duplicate scip symbols, indicating an internal rust-analyzer bug. These duplicates are
included in the output, but this causes information lookup to be ambiguous and so information about
these symbols presented by downstream tools may be incorrect.

Known rust-analyzer bugs that can cause this:

  * Definitions in crate example binaries which have the same symbol as definitions in the library
    or some other example.

  * Struct/enum/const/static/impl definitions nested in a function do not mention the function name.
    See #18771.

Duplicate symbols encountered:
";

fn compute_symbol_info(
    symbol: String,
    enclosing_symbol: Option<String>,
    token: &TokenStaticData,
) -> SymbolInformation {
    let documentation = match &token.documentation {
        Some(doc) => vec![doc.as_str().to_owned()],
        None => vec![],
    };

    let position_encoding = scip_types::PositionEncoding::UTF8CodeUnitOffsetFromLineStart.into();
    let signature_documentation = token.signature.clone().map(|text| scip_types::Document {
        relative_path: "".to_owned(),
        language: "rust".to_owned(),
        text,
        position_encoding,
        ..Default::default()
    });
    scip_types::SymbolInformation {
        symbol,
        documentation,
        relationships: Vec::new(),
        special_fields: Default::default(),
        kind: symbol_kind(token.kind).into(),
        display_name: token.display_name.clone().unwrap_or_default(),
        signature_documentation: signature_documentation.into(),
        enclosing_symbol: enclosing_symbol.unwrap_or_default(),
    }
}

fn get_relative_filepath(
    vfs: &vfs::Vfs,
    rootpath: &vfs::AbsPathBuf,
    file_id: ide::FileId,
) -> Option<String> {
    Some(vfs.file_path(file_id).as_path()?.strip_prefix(rootpath)?.as_str().to_owned())
}

fn get_line_index(db: &RootDatabase, file_id: FileId) -> LineIndex {
    LineIndex {
        index: db.line_index(file_id),
        encoding: PositionEncoding::Utf8,
        endings: LineEndings::Unix,
    }
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
    /// True if this symbol is for an inherent impl. This is used to only emit `SymbolInformation`
    /// for a struct's first inherent impl, since their symbol names are not disambiguated.
    is_inherent_impl: bool,
}

#[derive(Default)]
struct SymbolGenerator {
    token_to_symbols: FxHashMap<TokenId, Option<TokenSymbols>>,
    local_count: usize,
}

impl SymbolGenerator {
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
                        is_inherent_impl: match &moniker.identifier.description[..] {
                            // inherent impls are represented as impl#[SelfType]
                            [.., descriptor, _] => {
                                descriptor.desc == MonikerDescriptorKind::Type
                                    && descriptor.name == "impl"
                            }
                            _ => false,
                        },
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
                            is_inherent_impl: false,
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

    fn position(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> (AnalysisHost, FilePosition) {
        let mut host = AnalysisHost::default();
        let change_fixture = ChangeFixture::parse(host.raw_database(), ra_fixture);
        host.raw_database_mut().apply_change(change_fixture.change);
        let (file_id, range_or_offset) =
            change_fixture.file_position.expect("expected a marker ()");
        let offset = range_or_offset.expect_offset();
        let position = FilePosition { file_id: file_id.file_id(host.raw_database()), offset };
        (host, position)
    }

    /// If expected == "", then assert that there are no symbols (this is basically local symbol)
    #[track_caller]
    fn check_symbol(#[rust_analyzer::rust_fixture] ra_fixture: &str, expected: &str) {
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
                // check if cursor is within token, ignoring token for the module defined by the file (whose range is the whole file)
                if range.start() != TextSize::from(0) && range.contains(offset - TextSize::from(1))
                {
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

    // FIXME: This test represents current misbehavior.
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
            // FIXME: This should be a local:
            // "local enclosed by rust-analyzer cargo main . func().",
        );
    }

    // FIXME: This test represents current misbehavior.
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
            // FIXME: This should be a local:
            // "local enclosed by rust-analyzer cargo main . func().",
        );
    }

    // FIXME: This test represents current misbehavior.
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
            // FIXME: This should be a local:
            // "local enclosed by rust-analyzer cargo main . func().",
        );
    }

    // FIXME: This test represents current misbehavior.
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
            // FIXME: This should be a local:
            // "local enclosed by rust-analyzer cargo main . func().",
        );
    }

    #[test]
    fn documentation_matches_doc_comment() {
        let s = "/// foo\nfn bar() {}";

        let mut host = AnalysisHost::default();
        let change_fixture = ChangeFixture::parse(host.raw_database(), s);
        host.raw_database_mut().apply_change(change_fixture.change);

        let analysis = host.analysis();
        let si = StaticIndex::compute(
            &analysis,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );

        let file = si.files.first().unwrap();
        let (_, token_id) = file.tokens.get(1).unwrap(); // first token is file module, second is `bar`
        let token = si.tokens.get(*token_id).unwrap();

        assert_eq!(token.documentation.as_ref().map(|d| d.as_str()), Some("foo"));
    }
}
