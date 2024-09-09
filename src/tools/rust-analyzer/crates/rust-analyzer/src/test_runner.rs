//! This module provides the functionality needed to run `cargo test` in a background
//! thread and report the result of each test in a channel.

use std::process::Command;

use crossbeam_channel::Sender;
use paths::AbsPath;
use serde::Deserialize;
use toolchain::Tool;

use crate::{
    command::{CommandHandle, ParseFromLine},
    flycheck::CargoOptions,
};

#[derive(Debug, Deserialize)]
#[serde(tag = "event", rename_all = "camelCase")]
pub(crate) enum TestState {
    Started,
    Ok,
    Ignored,
    Failed { stdout: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub(crate) enum CargoTestMessage {
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

impl ParseFromLine for CargoTestMessage {
    fn from_line(line: &str, _: &mut String) -> Option<Self> {
        let mut deserializer = serde_json::Deserializer::from_str(line);
        deserializer.disable_recursion_limit();
        if let Ok(message) = CargoTestMessage::deserialize(&mut deserializer) {
            return Some(message);
        }

        Some(CargoTestMessage::Custom { text: line.to_owned() })
    }

    fn from_eof() -> Option<Self> {
        Some(CargoTestMessage::Finished)
    }
}

#[derive(Debug)]
pub(crate) struct CargoTestHandle {
    _handle: CommandHandle<CargoTestMessage>,
}

// Example of a cargo test command:
// cargo test --workspace --no-fail-fast -- -Z unstable-options --format=json
// or
// cargo test --package my-package --no-fail-fast -- module::func -Z unstable-options --format=json

#[derive(Debug)]
pub(crate) enum TestTarget {
    Workspace,
    Package(String),
}

impl CargoTestHandle {
    pub(crate) fn new(
        path: Option<&str>,
        options: CargoOptions,
        root: &AbsPath,
        test_target: TestTarget,
        sender: Sender<CargoTestMessage>,
    ) -> std::io::Result<Self> {
        let mut cmd = Command::new(Tool::Cargo.path());
        cmd.env("RUSTC_BOOTSTRAP", "1");
        cmd.arg("test");

        match &test_target {
            TestTarget::Package(package) => {
                cmd.arg("--package");
                cmd.arg(package);
            }
            TestTarget::Workspace => {
                cmd.arg("--workspace");
            }
        };

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

        Ok(Self { _handle: CommandHandle::spawn(cmd, sender)? })
    }
}
