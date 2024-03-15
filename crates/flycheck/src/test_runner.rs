//! This module provides the functionality needed to run `cargo test` in a background
//! thread and report the result of each test in a channel.

use std::process::Command;

use crossbeam_channel::Receiver;
use serde::Deserialize;
use toolchain::Tool;

use crate::command::{CommandHandle, ParseFromLine};

#[derive(Debug, Deserialize)]
#[serde(tag = "event", rename_all = "camelCase")]
pub enum TestState {
    Started,
    Ok,
    Ignored,
    Failed { stdout: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum CargoTestMessage {
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
pub struct CargoTestHandle {
    handle: CommandHandle<CargoTestMessage>,
}

// Example of a cargo test command:
// cargo test -- module::func -Z unstable-options --format=json

impl CargoTestHandle {
    pub fn new(path: Option<&str>) -> std::io::Result<Self> {
        let mut cmd = Command::new(Tool::Cargo.path());
        cmd.env("RUSTC_BOOTSTRAP", "1");
        cmd.arg("test");
        cmd.arg("--");
        if let Some(path) = path {
            cmd.arg(path);
        }
        cmd.args(["-Z", "unstable-options"]);
        cmd.arg("--format=json");
        Ok(Self { handle: CommandHandle::spawn(cmd)? })
    }

    pub fn receiver(&self) -> &Receiver<CargoTestMessage> {
        &self.handle.receiver
    }
}
