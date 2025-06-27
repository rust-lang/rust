use std::path::Path;
use std::process::Command;
use std::time::{Duration, SystemTime};
use std::{env, thread};

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

    loop {
        let index_time = mtime("util/gh-pages/index.html");
        let times = [
            "clippy_lints/src",
            "util/gh-pages/index_template.html",
            "tests/compile-test.rs",
        ]
        .map(mtime);

        if times.iter().any(|&time| index_time < time) {
            Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
                .arg("collect-metadata")
                .spawn()
                .unwrap()
                .wait()
                .unwrap();
        }
        if let Some(url) = url.take() {
            thread::spawn(move || {
                let mut child = Command::new(PYTHON)
                    .arg("-m")
                    .arg("http.server")
                    .arg(port.to_string())
                    .current_dir("util/gh-pages")
                    .spawn()
                    .unwrap();
                // Give some time for python to start
                thread::sleep(Duration::from_millis(500));
                // Launch browser after first export.py has completed and http.server is up
                let _result = opener::open(url);
                child.wait().unwrap();
            });
        }
        thread::sleep(Duration::from_millis(1000));
    }
}

fn mtime(path: impl AsRef<Path>) -> SystemTime {
    let path = path.as_ref();
    if path.is_dir() {
        path.read_dir()
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| mtime(entry.path()))
            .max()
            .unwrap_or(SystemTime::UNIX_EPOCH)
    } else {
        path.metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    }
}
