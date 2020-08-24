//! utils used in proc-macro tests

use crate::dylib;
use crate::ProcMacroSrv;
use proc_macro_api::ListMacrosTask;
use std::str::FromStr;
use test_utils::assert_eq_text;

mod fixtures {
    use cargo_metadata::Message;
    use std::process::Command;

    // Use current project metadata to get the proc-macro dylib path
    pub fn dylib_path(crate_name: &str, version: &str) -> std::path::PathBuf {
        let command = Command::new(toolchain::cargo())
            .args(&["check", "--tests", "--message-format", "json"])
            .output()
            .unwrap()
            .stdout;

        for message in Message::parse_stream(command.as_slice()) {
            match message.unwrap() {
                Message::CompilerArtifact(artifact) => {
                    if artifact.target.kind.contains(&"proc-macro".to_string()) {
                        let repr = format!("{} {}", crate_name, version);
                        if artifact.package_id.repr.starts_with(&repr) {
                            return artifact.filenames[0].clone();
                        }
                    }
                }
                _ => (), // Unknown message
            }
        }

        panic!("No proc-macro dylib for {} found!", crate_name);
    }
}

fn parse_string(code: &str) -> Option<crate::rustc_server::TokenStream> {
    Some(crate::rustc_server::TokenStream::from_str(code).unwrap())
}

pub fn assert_expand(
    crate_name: &str,
    macro_name: &str,
    version: &str,
    ra_fixture: &str,
    expect: &str,
) {
    let path = fixtures::dylib_path(crate_name, version);
    let expander = dylib::Expander::new(&path).unwrap();
    let fixture = parse_string(ra_fixture).unwrap();

    let res = expander.expand(macro_name, &fixture.subtree, None).unwrap();
    assert_eq_text!(&format!("{:?}", res), &expect.trim());
}

pub fn list(crate_name: &str, version: &str) -> Vec<String> {
    let path = fixtures::dylib_path(crate_name, version);
    let task = ListMacrosTask { lib: path };
    let mut srv = ProcMacroSrv::default();
    let res = srv.list_macros(&task).unwrap();
    res.macros.into_iter().map(|(name, kind)| format!("{} [{:?}]", name, kind)).collect()
}
