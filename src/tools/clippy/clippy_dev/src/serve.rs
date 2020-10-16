use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::{Duration, SystemTime};

pub fn run(port: u16, lint: Option<&str>) -> ! {
    let mut url = Some(match lint {
        None => format!("http://localhost:{}", port),
        Some(lint) => format!("http://localhost:{}/#{}", port, lint),
    });

    loop {
        if mtime("util/gh-pages/lints.json") < mtime("clippy_lints/src") {
            Command::new("python3")
                .arg("util/export.py")
                .spawn()
                .unwrap()
                .wait()
                .unwrap();
        }
        if let Some(url) = url.take() {
            thread::spawn(move || {
                Command::new("python3")
                    .arg("-m")
                    .arg("http.server")
                    .arg(port.to_string())
                    .current_dir("util/gh-pages")
                    .spawn()
                    .unwrap();
                // Give some time for python to start
                thread::sleep(Duration::from_millis(500));
                // Launch browser after first export.py has completed and http.server is up
                let _ = opener::open(url);
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
            .map(|entry| mtime(&entry.path()))
            .max()
            .unwrap_or(SystemTime::UNIX_EPOCH)
    } else {
        path.metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    }
}

#[allow(clippy::missing_errors_doc)]
pub fn validate_port(arg: &OsStr) -> Result<(), OsString> {
    match arg.to_string_lossy().parse::<u16>() {
        Ok(_port) => Ok(()),
        Err(err) => Err(OsString::from(err.to_string())),
    }
}
