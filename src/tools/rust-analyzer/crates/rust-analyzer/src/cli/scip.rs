//! SCIP generator

use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use crate::{
    cli::load_cargo::ProcMacroServerChoice,
    line_index::{LineEndings, LineIndex, PositionEncoding},
};
use hir::Name;
use ide::{
    LineCol, MonikerDescriptorKind, StaticIndex, StaticIndexedFile, TextRange, TokenId,
    TokenStaticData,
};
use ide_db::LineIndexDatabase;
use project_model::{CargoConfig, ProjectManifest, ProjectWorkspace, RustLibSource};
use scip::types as scip_types;
use std::env;

use crate::cli::{
    flags,
    load_cargo::{load_workspace, LoadCargoConfig},
    Result,
};

impl flags::Scip {
    pub fn run(self) -> Result<()> {
        eprintln!("Generating SCIP start...");
        let now = Instant::now();
        let mut cargo_config = CargoConfig::default();
        cargo_config.sysroot = Some(RustLibSource::Discover);

        let no_progress = &|s| (eprintln!("rust-analyzer: Loading {s}"));
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: true,
            with_proc_macro_server: ProcMacroServerChoice::Sysroot,
            prefill_caches: true,
        };
        let path = vfs::AbsPathBuf::assert(env::current_dir()?.join(&self.path));
        let rootpath = path.normalize();
        let manifest = ProjectManifest::discover_single(&path)?;

        let workspace = ProjectWorkspace::load(manifest, &cargo_config, no_progress)?;

        let (host, vfs, _) =
            load_workspace(workspace, &cargo_config.extra_env, &load_cargo_config)?;
        let db = host.raw_database();
        let analysis = host.analysis();

        let si = StaticIndex::compute(&analysis);

        let metadata = scip_types::Metadata {
            version: scip_types::ProtocolVersion::UnspecifiedProtocolVersion.into(),
            tool_info: Some(scip_types::ToolInfo {
                name: "rust-analyzer".to_owned(),
                version: "0.1".to_owned(),
                arguments: vec![],
                special_fields: Default::default(),
            })
            .into(),
            project_root: format!(
                "file://{}",
                path.normalize()
                    .as_os_str()
                    .to_str()
                    .ok_or(anyhow::anyhow!("Unable to normalize project_root path"))?
                    .to_string()
            ),
            text_document_encoding: scip_types::TextEncoding::UTF8.into(),
            special_fields: Default::default(),
        };
        let mut documents = Vec::new();

        let mut symbols_emitted: HashSet<TokenId> = HashSet::default();
        let mut tokens_to_symbol: HashMap<TokenId, String> = HashMap::new();

        for StaticIndexedFile { file_id, tokens, .. } in si.files {
            let mut local_count = 0;
            let mut new_local_symbol = || {
                let new_symbol = scip::types::Symbol::new_local(local_count);
                local_count += 1;

                new_symbol
            };

            let relative_path = match get_relative_filepath(&vfs, &rootpath, file_id) {
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

                let range = text_range_to_scip_range(&line_index, text_range);
                let symbol = tokens_to_symbol
                    .entry(id)
                    .or_insert_with(|| {
                        let symbol = token_to_symbol(token).unwrap_or_else(&mut new_local_symbol);
                        scip::symbol::format_symbol(symbol)
                    })
                    .clone();

                let mut symbol_roles = Default::default();

                if let Some(def) = token.definition {
                    if def.range == text_range {
                        symbol_roles |= scip_types::SymbolRole::Definition as i32;
                    }

                    if symbols_emitted.insert(id) {
                        let documentation = token
                            .hover
                            .as_ref()
                            .map(|hover| hover.markup.as_str())
                            .filter(|it| !it.is_empty())
                            .map(|it| vec![it.to_owned()]);
                        let symbol_info = scip_types::SymbolInformation {
                            symbol: symbol.clone(),
                            documentation: documentation.unwrap_or_default(),
                            relationships: Vec::new(),
                            special_fields: Default::default(),
                        };

                        symbols.push(symbol_info)
                    }
                }

                occurrences.push(scip_types::Occurrence {
                    range,
                    symbol,
                    symbol_roles,
                    override_documentation: Vec::new(),
                    syntax_kind: Default::default(),
                    diagnostics: Vec::new(),
                    special_fields: Default::default(),
                });
            });

            if occurrences.is_empty() {
                continue;
            }

            documents.push(scip_types::Document {
                relative_path,
                language: "rust".to_string(),
                occurrences,
                symbols,
                special_fields: Default::default(),
            });
        }

        let index = scip_types::Index {
            metadata: Some(metadata).into(),
            documents,
            external_symbols: Vec::new(),
            special_fields: Default::default(),
        };

        scip::write_message_to_file("index.scip", index)
            .map_err(|err| anyhow::anyhow!("Failed to write scip to file: {}", err))?;

        eprintln!("Generating SCIP finished {:?}", now.elapsed());
        Ok(())
    }
}

fn get_relative_filepath(
    vfs: &vfs::Vfs,
    rootpath: &vfs::AbsPathBuf,
    file_id: ide::FileId,
) -> Option<String> {
    Some(vfs.file_path(file_id).as_path()?.strip_prefix(rootpath)?.as_ref().to_str()?.to_string())
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

fn new_descriptor_str(
    name: &str,
    suffix: scip_types::descriptor::Suffix,
) -> scip_types::Descriptor {
    scip_types::Descriptor {
        name: name.to_string(),
        disambiguator: "".to_string(),
        suffix: suffix.into(),
        special_fields: Default::default(),
    }
}

fn new_descriptor(name: Name, suffix: scip_types::descriptor::Suffix) -> scip_types::Descriptor {
    let mut name = name.to_string();
    if name.contains("'") {
        name = format!("`{name}`");
    }

    new_descriptor_str(name.as_str(), suffix)
}

/// Loosely based on `def_to_moniker`
///
/// Only returns a Symbol when it's a non-local symbol.
///     So if the visibility isn't outside of a document, then it will return None
fn token_to_symbol(token: &TokenStaticData) -> Option<scip_types::Symbol> {
    use scip_types::descriptor::Suffix::*;

    let moniker = token.moniker.as_ref()?;

    let package_name = moniker.package_information.name.clone();
    let version = moniker.package_information.version.clone();
    let descriptors = moniker
        .identifier
        .description
        .iter()
        .map(|desc| {
            new_descriptor(
                desc.name.clone(),
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
        .collect();

    Some(scip_types::Symbol {
        scheme: "rust-analyzer".into(),
        package: Some(scip_types::Package {
            manager: "cargo".to_string(),
            name: package_name,
            version: version.unwrap_or_else(|| ".".to_string()),
            special_fields: Default::default(),
        })
        .into(),
        descriptors,
        special_fields: Default::default(),
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use ide::{AnalysisHost, FilePosition, StaticIndex, TextSize};
    use ide_db::base_db::fixture::ChangeFixture;
    use scip::symbol::format_symbol;

    fn position(ra_fixture: &str) -> (AnalysisHost, FilePosition) {
        let mut host = AnalysisHost::default();
        let change_fixture = ChangeFixture::parse(ra_fixture);
        host.raw_database_mut().apply_change(change_fixture.change);
        let (file_id, range_or_offset) =
            change_fixture.file_position.expect("expected a marker ($0)");
        let offset = range_or_offset.expect_offset();
        (host, FilePosition { file_id, offset })
    }

    /// If expected == "", then assert that there are no symbols (this is basically local symbol)
    #[track_caller]
    fn check_symbol(ra_fixture: &str, expected: &str) {
        let (host, position) = position(ra_fixture);

        let analysis = host.analysis();
        let si = StaticIndex::compute(&analysis);

        let FilePosition { file_id, offset } = position;

        let mut found_symbol = None;
        for file in &si.files {
            if file.file_id != file_id {
                continue;
            }
            for &(range, id) in &file.tokens {
                if range.contains(offset - TextSize::from(1)) {
                    let token = si.tokens.get(id).unwrap();
                    found_symbol = token_to_symbol(token);
                    break;
                }
            }
        }

        if expected == "" {
            assert!(found_symbol.is_none(), "must have no symbols {found_symbol:?}");
            return;
        }

        assert!(found_symbol.is_some(), "must have one symbol {found_symbol:?}");
        let res = found_symbol.unwrap();
        let formatted = format_symbol(res);
        assert_eq!(formatted, expected);
    }

    #[test]
    fn basic() {
        check_symbol(
            r#"
//- /lib.rs crate:main deps:foo
use foo::example_mod::func;
fn main() {
    func$0();
}
//- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
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
//- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
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
    fn symbol_for_trait_constant() {
        check_symbol(
            r#"
    //- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
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
    //- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
    pub mod module {
        pub trait MyTrait {
            type MyType$0;
        }
    }
    "#,
            // "foo::module::MyTrait::MyType",
            "rust-analyzer cargo foo 0.1.0 module/MyTrait#[MyType]",
        );
    }

    #[test]
    fn symbol_for_trait_impl_function() {
        check_symbol(
            r#"
    //- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
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
            // "foo::module::MyStruct::MyTrait::func",
            "rust-analyzer cargo foo 0.1.0 module/MyStruct#MyTrait#func().",
        );
    }

    #[test]
    fn symbol_for_field() {
        check_symbol(
            r#"
    //- /lib.rs crate:main deps:foo
    use foo::St;
    fn main() {
        let x = St { a$0: 2 };
    }
    //- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
    pub struct St {
        pub a: i32,
    }
    "#,
            "rust-analyzer cargo foo 0.1.0 St#a.",
        );
    }

    #[test]
    fn local_symbol_for_local() {
        check_symbol(
            r#"
    //- /lib.rs crate:main deps:foo
    use foo::module::func;
    fn main() {
        func();
    }
    //- /foo/lib.rs crate:foo@CratesIo:0.1.0,https://a.b/foo.git
    pub mod module {
        pub fn func() {
            let x$0 = 2;
        }
    }
    "#,
            "",
        );
    }

    #[test]
    fn global_symbol_for_pub_struct() {
        check_symbol(
            r#"
    //- /lib.rs crate:main
    mod foo;

    fn main() {
        let _bar = foo::Bar { i: 0 };
    }
    //- /foo.rs
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
    //- /lib.rs crate:main
    mod foo;

    fn main() {
        let _bar = foo::Bar$0 { i: 0 };
    }
    //- /foo.rs
    pub struct Bar {
        pub i: i32,
    }
    "#,
            "rust-analyzer cargo main . foo/Bar#",
        );
    }
}
