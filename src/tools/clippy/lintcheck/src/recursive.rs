//! In `--recursive` mode we set the `lintcheck` binary as the `RUSTC_WRAPPER` of `cargo check`,
//! this allows [`crate::driver`] to be run for every dependency. The driver connects to
//! [`LintcheckServer`] to ask if it should be skipped, and if not sends the stderr of running
//! clippy on the crate to the server

use crate::ClippyWarning;
use crate::input::RecursiveOptions;

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use cargo_metadata::diagnostic::Diagnostic;
use crossbeam_channel::{Receiver, Sender};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Debug, Eq, Hash, PartialEq, Clone, Serialize, Deserialize)]
pub(crate) struct DriverInfo {
    pub package_name: String,
    pub version: String,
}

pub(crate) fn serialize_line<T, W>(value: &T, writer: &mut W)
where
    T: Serialize,
    W: Write,
{
    let mut buf = serde_json::to_vec(&value).expect("failed to serialize");
    buf.push(b'\n');
    writer.write_all(&buf).expect("write_all failed");
}

pub(crate) fn deserialize_line<T, R>(reader: &mut R) -> T
where
    T: DeserializeOwned,
    R: BufRead,
{
    let mut string = String::new();
    reader.read_line(&mut string).expect("read_line failed");
    serde_json::from_str(&string).expect("failed to deserialize")
}

fn process_stream(
    stream: TcpStream,
    sender: &Sender<ClippyWarning>,
    options: &RecursiveOptions,
    seen: &Mutex<HashSet<DriverInfo>>,
) {
    let mut stream = BufReader::new(stream);

    let driver_info: DriverInfo = deserialize_line(&mut stream);

    let unseen = seen.lock().unwrap().insert(driver_info.clone());
    let ignored = options.ignore.contains(&driver_info.package_name);
    let should_run = unseen && !ignored;

    serialize_line(&should_run, stream.get_mut());

    let mut stderr = String::new();
    stream.read_to_string(&mut stderr).unwrap();

    // It's 99% likely that dependencies compiled with recursive mode are on crates.io
    // and therefore on docs.rs. This links to the sources directly, do avoid invalid
    // links due to remapped paths. See rust-lang/docs.rs#2551 for more details.
    let base_url = format!(
        "https://docs.rs/crate/{}/{}/source/src/{{file}}#{{line}}",
        driver_info.package_name, driver_info.version
    );
    let messages = stderr
        .lines()
        .filter_map(|json_msg| serde_json::from_str::<Diagnostic>(json_msg).ok())
        .filter_map(|diag| ClippyWarning::new(diag, &base_url, &driver_info.package_name));

    for message in messages {
        sender.send(message).unwrap();
    }
}

pub(crate) struct LintcheckServer {
    pub local_addr: SocketAddr,
    receiver: Receiver<ClippyWarning>,
    sender: Arc<Sender<ClippyWarning>>,
}

impl LintcheckServer {
    pub fn spawn(options: RecursiveOptions) -> Self {
        let listener = TcpListener::bind("localhost:0").unwrap();
        let local_addr = listener.local_addr().unwrap();

        let (sender, receiver) = crossbeam_channel::unbounded::<ClippyWarning>();
        let sender = Arc::new(sender);
        // The spawned threads hold a `Weak<Sender>` so that they don't keep the channel connected
        // indefinitely
        let sender_weak = Arc::downgrade(&sender);

        // Ignore dependencies multiple times, e.g. for when it's both checked and compiled for a
        // build dependency
        let seen = Mutex::default();

        thread::spawn(move || {
            thread::scope(|s| {
                s.spawn(|| {
                    while let Ok((stream, _)) = listener.accept() {
                        let sender = sender_weak.upgrade().expect("received connection after server closed");
                        let options = &options;
                        let seen = &seen;
                        s.spawn(move || process_stream(stream, &sender, options, seen));
                    }
                });
            });
        });

        Self {
            local_addr,
            receiver,
            sender,
        }
    }

    pub fn warnings(self) -> impl Iterator<Item = ClippyWarning> {
        // causes the channel to become disconnected so that the receiver iterator ends
        drop(self.sender);

        self.receiver.into_iter()
    }
}
