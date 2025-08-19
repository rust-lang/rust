//! This module provides the functionality needed to run `cargo test` in a background
//! thread and report the result of each test in a channel.

use crossbeam_channel::Sender;
use paths::AbsPath;
use project_model::TargetKind;
use serde::Deserialize as _;
use serde_derive::Deserialize;
use toolchain::Tool;

use crate::{
    command::{CargoParser, CommandHandle},
    flycheck::CargoOptions,
};

#[derive(Debug, Deserialize)]
#[serde(tag = "event", rename_all = "camelCase")]
pub(crate) enum TestState {
    Started,
    Ok,
    Ignored,
    Failed {
        // the stdout field is not always present depending on cargo test flags
        #[serde(skip_serializing_if = "String::is_empty", default)]
        stdout: String,
    },
}

#[derive(Debug)]
pub(crate) struct CargoTestMessage {
    pub target: TestTarget,
    pub output: CargoTestOutput,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub(crate) enum CargoTestOutput {
    Test {
        name: String,
        #[serde(flatten)]
        state: TestState,
    },
    Suite,
    Finished,
    Custom {
        text: String,
    },
}

pub(crate) struct CargoTestOutputParser {
    pub target: TestTarget,
}

impl CargoTestOutputParser {
    pub(crate) fn new(test_target: &TestTarget) -> Self {
        Self { target: test_target.clone() }
    }
}

impl CargoParser<CargoTestMessage> for CargoTestOutputParser {
    fn from_line(&self, line: &str, _error: &mut String) -> Option<CargoTestMessage> {
        let mut deserializer = serde_json::Deserializer::from_str(line);
        deserializer.disable_recursion_limit();

        Some(CargoTestMessage {
            target: self.target.clone(),
            output: if let Ok(message) = CargoTestOutput::deserialize(&mut deserializer) {
                message
            } else {
                CargoTestOutput::Custom { text: line.to_owned() }
            },
        })
    }

    fn from_eof(&self) -> Option<CargoTestMessage> {
        Some(CargoTestMessage { target: self.target.clone(), output: CargoTestOutput::Finished })
    }
}

#[derive(Debug)]
pub(crate) struct CargoTestHandle {
    _handle: CommandHandle<CargoTestMessage>,
}

// Example of a cargo test command:
//
// cargo test --package my-package --bin my_bin --no-fail-fast -- module::func -Z unstable-options --format=json

#[derive(Debug, Clone)]
pub(crate) struct TestTarget {
    pub package: String,
    pub target: String,
    pub kind: TargetKind,
}

impl CargoTestHandle {
    pub(crate) fn new(
        path: Option<&str>,
        options: CargoOptions,
        root: &AbsPath,
        test_target: TestTarget,
        sender: Sender<CargoTestMessage>,
    ) -> std::io::Result<Self> {
        let mut cmd = toolchain::command(Tool::Cargo.path(), root, &options.extra_env);
        cmd.env("RUSTC_BOOTSTRAP", "1");
        cmd.arg("--color=always");
        cmd.arg("test");

        cmd.arg("--package");
        cmd.arg(&test_target.package);

        if let TargetKind::Lib { .. } = test_target.kind {
            // no name required with lib because there can only be one lib target per package
            cmd.arg("--lib");
        } else if let Some(cargo_target) = test_target.kind.as_cargo_target() {
            cmd.arg(format!("--{cargo_target}"));
            cmd.arg(&test_target.target);
        } else {
            tracing::warn!("Running test for unknown cargo target {:?}", test_target.kind);
        }

        // --no-fail-fast is needed to ensure that all requested tests will run
        cmd.arg("--no-fail-fast");
        cmd.arg("--manifest-path");
        cmd.arg(root.join("Cargo.toml"));
        options.apply_on_command(&mut cmd);
        cmd.arg("--");
        if let Some(path) = path {
            cmd.arg(path);
        }
        cmd.args(["-Z", "unstable-options"]);
        cmd.arg("--format=json");

        for extra_arg in options.extra_test_bin_args {
            cmd.arg(extra_arg);
        }

        Ok(Self {
            _handle: CommandHandle::spawn(cmd, CargoTestOutputParser::new(&test_target), sender)?,
        })
    }
}
