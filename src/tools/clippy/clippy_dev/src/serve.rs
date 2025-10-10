use crate::utils::{ErrAction, cargo_cmd, expect_action};
use core::fmt::Display;
use core::mem;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, SystemTime};
use std::{fs, thread};
use walkdir::WalkDir;

#[cfg(windows)]
const PYTHON: &str = "python";

#[cfg(not(windows))]
const PYTHON: &str = "python3";

/// # Panics
///
/// Panics if the python commands could not be spawned
pub fn run(port: u16, lint: Option<String>) -> ! {
    let mut url = Some(match lint {
        None => format!("http://localhost:{port}"),
        Some(lint) => format!("http://localhost:{port}/#{lint}"),
    });

    let mut last_update = mtime("util/gh-pages/index.html");
    loop {
        if is_metadata_outdated(mem::replace(&mut last_update, SystemTime::now())) {
            // Ignore the command result; we'll fall back to displaying the old metadata.
            let _ = expect_action(
                cargo_cmd().arg("collect-metadata").status(),
                ErrAction::Run,
                "cargo collect-metadata",
            );
            last_update = SystemTime::now();
        }

        // Only start the web server the first time around.
        if let Some(url) = url.take() {
            thread::spawn(move || {
                let mut child = expect_action(
                    Command::new(PYTHON)
                        .args(["-m", "http.server", port.to_string().as_str()])
                        .current_dir("util/gh-pages")
                        .spawn(),
                    ErrAction::Run,
                    "python -m http.server",
                );
                // Give some time for python to start
                thread::sleep(Duration::from_millis(500));
                // Launch browser after first export.py has completed and http.server is up
                let _result = opener::open(url);
                expect_action(child.wait(), ErrAction::Run, "python -m http.server");
            });
        }

        // Delay to avoid updating the metadata too aggressively.
        thread::sleep(Duration::from_millis(1000));
    }
}

fn log_err_and_continue<T>(res: Result<T, impl Display>, path: &Path) -> Option<T> {
    match res {
        Ok(x) => Some(x),
        Err(ref e) => {
            eprintln!("error reading `{}`: {e}", path.display());
            None
        },
    }
}

fn mtime(path: &str) -> SystemTime {
    log_err_and_continue(fs::metadata(path), path.as_ref())
        .and_then(|metadata| log_err_and_continue(metadata.modified(), path.as_ref()))
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

fn is_metadata_outdated(time: SystemTime) -> bool {
    // Ignore all IO errors here. We don't want to stop them from hosting the server.
    if time < mtime("util/gh-pages/index_template.html") || time < mtime("tests/compile-test.rs") {
        return true;
    }
    let Some(dir) = log_err_and_continue(fs::read_dir("."), ".".as_ref()) else {
        return false;
    };
    dir.map_while(|e| log_err_and_continue(e, ".".as_ref())).any(|e| {
        let name = e.file_name();
        let name_bytes = name.as_encoded_bytes();
        if (name_bytes.starts_with(b"clippy_lints") && name_bytes != b"clippy_lints_internal")
            || name_bytes == b"clippy_config"
        {
            WalkDir::new(&name)
                .into_iter()
                .map_while(|e| log_err_and_continue(e, name.as_ref()))
                .filter(|e| e.file_type().is_file())
                .filter_map(|e| {
                    log_err_and_continue(e.metadata(), e.path())
                        .and_then(|m| log_err_and_continue(m.modified(), e.path()))
                })
                .any(|ftime| time < ftime)
        } else {
            false
        }
    })
}
