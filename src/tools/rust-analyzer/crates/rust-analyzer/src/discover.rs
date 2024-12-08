//! Infrastructure for lazy project discovery. Currently only support rust-project.json discovery
//! via a custom discover command.
use std::{io, process::Command};

use crossbeam_channel::Sender;
use paths::{AbsPathBuf, Utf8Path, Utf8PathBuf};
use project_model::ProjectJsonData;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info_span, span::EnteredSpan};

use crate::command::{CommandHandle, ParseFromLine};

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

    /// Spawn the command inside [Discover] and report progress, if any.
    pub(crate) fn spawn(&self, discover_arg: DiscoverArgument) -> io::Result<DiscoverHandle> {
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

        let mut cmd = Command::new(command);
        cmd.args(args);

        Ok(DiscoverHandle {
            _handle: CommandHandle::spawn(cmd, self.sender.clone())?,
            span: info_span!("discover_command").entered(),
        })
    }
}

/// A handle to a spawned [Discover].
#[derive(Debug)]
pub(crate) struct DiscoverHandle {
    _handle: CommandHandle<DiscoverProjectMessage>,
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

impl ParseFromLine for DiscoverProjectMessage {
    fn from_line(line: &str, _error: &mut String) -> Option<Self> {
        // can the line even be deserialized as JSON?
        let Ok(data) = serde_json::from_str::<Value>(line) else {
            let err = DiscoverProjectData::Error { error: line.to_owned(), source: None };
            return Some(DiscoverProjectMessage::new(err));
        };

        let Ok(data) = serde_json::from_value::<DiscoverProjectData>(data) else {
            return None;
        };

        let msg = DiscoverProjectMessage::new(data);
        Some(msg)
    }

    fn from_eof() -> Option<Self> {
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
