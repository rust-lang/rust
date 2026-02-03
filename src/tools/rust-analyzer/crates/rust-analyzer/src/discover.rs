//! Infrastructure for lazy project discovery. Currently only support rust-project.json discovery
//! via a custom discover command.
use std::path::Path;

use crossbeam_channel::Sender;
use ide_db::FxHashMap;
use paths::{AbsPathBuf, Utf8Path, Utf8PathBuf};
use project_model::ProjectJsonData;
use serde::{Deserialize, Serialize};
use tracing::{info_span, span::EnteredSpan};

use crate::command::{CommandHandle, JsonLinesParser};

pub(crate) const ARG_PLACEHOLDER: &str = "{arg}";

/// A command wrapper for getting a `rust-project.json`.
///
/// This is analogous to discovering a cargo project + running `cargo-metadata` on it, but for non-Cargo build systems.
pub(crate) struct DiscoverCommand {
    command: Vec<String>,
    sender: Sender<DiscoverProjectMessage>,
}

#[derive(PartialEq, Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum DiscoverArgument {
    Path(#[serde(serialize_with = "serialize_abs_pathbuf")] AbsPathBuf),
    Buildfile(#[serde(serialize_with = "serialize_abs_pathbuf")] AbsPathBuf),
}

fn serialize_abs_pathbuf<S>(path: &AbsPathBuf, se: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let path: &Utf8Path = path.as_ref();
    se.serialize_str(path.as_str())
}

impl DiscoverCommand {
    /// Create a new [DiscoverCommand].
    pub(crate) fn new(sender: Sender<DiscoverProjectMessage>, command: Vec<String>) -> Self {
        Self { sender, command }
    }

    /// Spawn the command inside `DiscoverCommand` and report progress, if any.
    pub(crate) fn spawn(
        &self,
        discover_arg: DiscoverArgument,
        current_dir: &Path,
    ) -> anyhow::Result<DiscoverHandle> {
        let command = &self.command[0];
        let args = &self.command[1..];

        let args: Vec<String> = args
            .iter()
            .map(|arg| {
                if arg == ARG_PLACEHOLDER {
                    serde_json::to_string(&discover_arg).expect("Unable to serialize args")
                } else {
                    arg.to_owned()
                }
            })
            .collect();

        // TODO: are we sure the extra env should be empty?
        let mut cmd = toolchain::command(command, current_dir, &FxHashMap::default());
        cmd.args(args);

        Ok(DiscoverHandle {
            handle: CommandHandle::spawn(cmd, DiscoverProjectParser, self.sender.clone(), None)?,
            span: info_span!("discover_command").entered(),
        })
    }
}

/// A handle to a spawned `DiscoverCommand`.
#[derive(Debug)]
pub(crate) struct DiscoverHandle {
    pub(crate) handle: CommandHandle<DiscoverProjectMessage>,
    #[allow(dead_code)] // not accessed, but used to log on drop.
    span: EnteredSpan,
}

/// An enum containing either progress messages, an error,
/// or the materialized `rust-project`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind")]
#[serde(rename_all = "snake_case")]
enum DiscoverProjectData {
    Finished { buildfile: Utf8PathBuf, project: ProjectJsonData },
    Error { error: String, source: Option<String> },
    Progress { message: String },
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum DiscoverProjectMessage {
    Finished { project: ProjectJsonData, buildfile: AbsPathBuf },
    Error { error: String, source: Option<String> },
    Progress { message: String },
}

impl DiscoverProjectMessage {
    fn new(data: DiscoverProjectData) -> Self {
        match data {
            DiscoverProjectData::Finished { project, buildfile, .. } => {
                let buildfile = buildfile.try_into().expect("Unable to make path absolute");
                DiscoverProjectMessage::Finished { project, buildfile }
            }
            DiscoverProjectData::Error { error, source } => {
                DiscoverProjectMessage::Error { error, source }
            }
            DiscoverProjectData::Progress { message } => {
                DiscoverProjectMessage::Progress { message }
            }
        }
    }
}

struct DiscoverProjectParser;

impl JsonLinesParser<DiscoverProjectMessage> for DiscoverProjectParser {
    fn from_line(&self, line: &str, _error: &mut String) -> Option<DiscoverProjectMessage> {
        match serde_json::from_str::<DiscoverProjectData>(line) {
            Ok(data) => {
                let msg = DiscoverProjectMessage::new(data);
                Some(msg)
            }
            Err(err) => {
                let err =
                    DiscoverProjectData::Error { error: format!("{err:#?}\n{line}"), source: None };
                Some(DiscoverProjectMessage::new(err))
            }
        }
    }

    fn from_eof(&self) -> Option<DiscoverProjectMessage> {
        None
    }
}

#[test]
fn test_deserialization() {
    let message = r#"
    {"kind": "progress", "message":"querying build system","input":{"files":["src/main.rs"]}}
    "#;
    let message: DiscoverProjectData =
        serde_json::from_str(message).expect("Unable to deserialize message");
    assert!(matches!(message, DiscoverProjectData::Progress { .. }));

    let message = r#"
    {"kind": "error", "error":"failed to deserialize command output","source":"command"}
    "#;

    let message: DiscoverProjectData =
        serde_json::from_str(message).expect("Unable to deserialize message");
    assert!(matches!(message, DiscoverProjectData::Error { .. }));

    let message = r#"
    {"kind": "finished", "project": {"sysroot": "foo", "crates": [], "runnables": []}, "buildfile":"rust-analyzer/BUILD"}
    "#;

    let message: DiscoverProjectData =
        serde_json::from_str(message).expect("Unable to deserialize message");
    assert!(matches!(message, DiscoverProjectData::Finished { .. }));
}
