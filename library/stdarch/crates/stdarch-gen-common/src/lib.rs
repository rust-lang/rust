//! Shared check/bless harness for stdarch generators.

use similar::TextDiff;
use std::error::Error as StdError;
use std::fmt;
use std::fs;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str::FromStr;

/// First-line marker identifying an auto-generated file. Generators emit this
/// as the first line of every file they produce; the harness uses it to
/// discover which files in `committed` are owned by the generator.
pub const GENERATED_MARKER: &str = "// This code is automatically generated. DO NOT MODIFY.";

/// Controls what `run_generator` does with the generator's output.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Verify that the `committed` matches the generator's output for owned files.
    ///
    /// Runs the generator into a temp directory, then compares each produced
    /// file against the committed copy. Returns an error on the first mismatch.
    Check,
    /// Update the `committed` to match the generator's output for owned files.
    ///
    /// Runs the generator into a temp directory and copies each produced file
    /// into `committed`. If the generator no longer produces an owned file, the
    /// committed copy is deleted. Files in `committed` that are not owned
    /// are left untouched.
    #[default]
    Bless,
}

impl FromStr for Mode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "check" => Ok(Mode::Check),
            "bless" => Ok(Mode::Bless),
            other => Err(format!(
                "unknown stdarch generation mode: {other:?}. Possible values are `check` or `bless`."
            )),
        }
    }
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Mismatch { path: PathBuf, kind: MismatchKind },
    Generator(Box<dyn StdError + Send + Sync>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MismatchKind {
    /// Owned file produced by the generator but absent from the `committed`.
    /// Means the `committed` needs to be regenerated.
    MissingInCommitted,
    /// Owned file present in the `committed` but the generator no longer
    /// produces it. The file must be removed from the `committed` .
    ExtraInCommitted,
    /// Owned file exists on both sides but contents differ.
    ContentsDiffer,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Mismatch { path, kind } => match kind {
                MismatchKind::MissingInCommitted => {
                    write!(f, "{}: generated but not committed", path.display())
                }
                MismatchKind::ExtraInCommitted => {
                    write!(f, "{}: committed but no longer generated", path.display())
                }
                MismatchKind::ContentsDiffer => write!(f, "{}: contents differ", path.display()),
            },
            Error::Generator(e) => write!(f, "generator failed: {e}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Generator(e) => Some(&**e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub struct GeneratorCtx {
    rustfmt_path: PathBuf,
}

impl GeneratorCtx {
    pub fn new(rustfmt_path: Option<PathBuf>) -> Self {
        Self {
            rustfmt_path: rustfmt_path.unwrap_or_else(|| PathBuf::from("rustfmt")),
        }
    }
}

/// Run a generator under the chosen `mode`, reconciling its output with `committed`.
///
/// Arguments:
/// - `committed` — the directory holding the in-tree (committed) source files.
///   Files inside `committed` whose first line begins with [`GENERATED_MARKER`]
///   are treated as owned by the generator. Anything else is treated as
///   hand-written and is left untouched by `Bless` and ignored by `Check`.
///   So generated files coexist with hand-written files in the same  directory.
/// - `mode` — what to do with the generator's output.
/// - `generate` — closure that writes the generator's output into the
///   directory it is given. Its error is wrapped in [`Error::Generator`].
///
/// Behavior per mode:
/// - [`Mode::Check`]: runs the generator into a temp dir and compares owned
///   files against the committed copies. Mismatch returns [`Error::Mismatch`].
/// - [`Mode::Bless`]: runs the generator into a temp dir and copies owned
///   files into `committed`, or removes `committed`'s copy if the generator no
///   longer produces them.
pub fn run_generator<F, E>(
    ctx: &GeneratorCtx,
    committed: &Path,
    mode: Mode,
    generate: F,
) -> Result<()>
where
    F: FnOnce(&Path) -> std::result::Result<(), E>,
    E: Into<Box<dyn StdError + Send + Sync>>,
{
    let scratch = tempfile::tempdir()?;
    generate(scratch.path()).map_err(|e| Error::Generator(e.into()))?;

    let owned = discover_owned(committed)?;
    let produced = discover_all(scratch.path())?;

    // Format all generated Rust files
    for file in &produced {
        let fullpath = scratch.path().join(file);
        if fullpath.extension().and_then(|s| s.to_str()) == Some("rs") {
            reformat_file(ctx, &fullpath)?;
        }
    }

    let mut names: Vec<&String> = owned.iter().chain(produced.iter()).collect();
    names.sort();
    names.dedup();

    for name in names {
        match mode {
            Mode::Check => compare(scratch.path(), committed, name)?,
            Mode::Bless => apply_bless(scratch.path(), committed, name)?,
        }
    }
    Ok(())
}

fn reformat_file(ctx: &GeneratorCtx, path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::open(path)?;
    let proc = Command::new(&ctx.rustfmt_path)
        // Ensure that rustfmt config files in other directories won't interfere with the formatting
        // This is important for usage within the rust-lang/rust repository
        .arg("--config-path")
        .arg(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("rustfmt.toml"),
        )
        .stdin(Stdio::from(file))
        .stdout(Stdio::piped())
        .spawn()?;

    let output = proc.wait_with_output()?;
    if !output.status.success() {
        panic!(
            "Running {:?} on {path:?} failed with exit code {:?}",
            ctx.rustfmt_path, output.status
        );
    }
    std::fs::write(path, output.stdout)
}

/// Returns the names of files in `dir` whose first line begins with
/// [`GENERATED_MARKER`]. Files without the marker are skipped.
fn discover_owned(dir: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let Ok(mut file) = fs::File::open(entry.path()) else {
            continue;
        };
        let mut buf = [0u8; GENERATED_MARKER.len()];
        if file.read_exact(&mut buf).is_err() {
            continue;
        }
        if buf == *GENERATED_MARKER.as_bytes()
            && let Some(name) = entry.file_name().to_str()
        {
            out.push(name.to_owned());
        }
    }
    out.sort();
    Ok(out)
}

/// Returns every file name in `dir` (scratch dir — all files are generator output).
fn discover_all(dir: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file()
            && let Some(name) = entry.file_name().to_str()
        {
            out.push(name.to_string());
        }
    }
    Ok(out)
}

fn compare(generated_dir: &Path, committed_dir: &Path, filename: &str) -> Result<()> {
    let rel_path = PathBuf::from(filename);
    let gen_path = generated_dir.join(&rel_path);
    let comm_path = committed_dir.join(&rel_path);
    match (gen_path.exists(), comm_path.exists()) {
        (true, false) => Err(Error::Mismatch {
            path: rel_path,
            kind: MismatchKind::MissingInCommitted,
        }),
        (false, true) => Err(Error::Mismatch {
            path: rel_path,
            kind: MismatchKind::ExtraInCommitted,
        }),
        (false, false) => Ok(()),
        (true, true) => {
            let generated = fs::read(&gen_path)?;
            let committed = fs::read(&comm_path)?;
            if generated != committed {
                if let (Ok(committed), Ok(generated)) =
                    (str::from_utf8(&committed), str::from_utf8(&generated))
                {
                    eprintln!(
                        "{}",
                        TextDiff::from_lines(committed, generated)
                            .unified_diff()
                            .context_radius(3)
                            .header(
                                &format!("committed/{filename}"),
                                &format!("generated/{filename}"),
                            )
                    );
                }
                Err(Error::Mismatch {
                    path: rel_path,
                    kind: MismatchKind::ContentsDiffer,
                })
            } else {
                Ok(())
            }
        }
    }
}

fn apply_bless(generated_dir: &Path, committed_dir: &Path, filename: &str) -> Result<()> {
    fs::create_dir_all(committed_dir)?;
    let rel_path = PathBuf::from(filename);
    let from = generated_dir.join(&rel_path);
    let to = committed_dir.join(&rel_path);
    if from.exists() {
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&from, &to)?;
    } else if to.exists() {
        fs::remove_file(&to)?;
    }
    Ok(())
}

// Skipped on iOS because these tests fail on the `x86_64-apple-ios-macabi` CI runner
// with `Os { code: 17, kind: AlreadyExists, message: "File exists" }`.
#[cfg(all(test, not(target_os = "ios")))]
mod tests {
    use super::*;

    fn write_marker(p: &Path, body: &[u8]) {
        if let Some(d) = p.parent() {
            fs::create_dir_all(d).unwrap();
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GENERATED_MARKER.as_bytes());
        bytes.push(b'\n');
        bytes.extend_from_slice(body);
        fs::write(p, bytes).unwrap();
    }

    #[test]
    fn check_fails_on_byte_diff() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        write_marker(&committed.join("a.txt"), b"hi");

        let ctx = GeneratorCtx::new(None);
        let e = run_generator(
            &ctx,
            &committed,
            Mode::Check,
            |out| -> std::result::Result<(), io::Error> {
                write_marker(&out.join("a.txt"), b"HI");
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(
            e,
            Error::Mismatch {
                kind: MismatchKind::ContentsDiffer,
                ..
            }
        ));
    }

    #[test]
    fn check_fails_when_owned_file_missing_from_generated() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        write_marker(&committed.join("a.txt"), b"hi");

        let ctx = GeneratorCtx::new(None);
        let e = run_generator(
            &ctx,
            &committed,
            Mode::Check,
            |_| -> std::result::Result<(), io::Error> { Ok(()) },
        )
        .unwrap_err();
        assert!(matches!(
            e,
            Error::Mismatch {
                kind: MismatchKind::ExtraInCommitted,
                ..
            }
        ));
    }

    #[test]
    fn check_fails_when_owned_file_missing_from_committed() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        fs::create_dir_all(&committed).unwrap();

        let ctx = GeneratorCtx::new(None);
        let e = run_generator(
            &ctx,
            &committed,
            Mode::Check,
            |out| -> std::result::Result<(), io::Error> {
                write_marker(&out.join("a.txt"), b"hi");
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(
            e,
            Error::Mismatch {
                kind: MismatchKind::MissingInCommitted,
                ..
            }
        ));
    }

    #[test]
    fn bless_deletes_files_no_longer_produced() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        write_marker(&committed.join("keep.txt"), b"");
        write_marker(&committed.join("stale.txt"), b"");

        let ctx = GeneratorCtx::new(None);
        run_generator(
            &ctx,
            &committed,
            Mode::Bless,
            |out| -> std::result::Result<(), io::Error> {
                write_marker(&out.join("keep.txt"), b"");
                Ok(())
            },
        )
        .unwrap();
        assert!(committed.join("keep.txt").exists());
        assert!(!committed.join("stale.txt").exists());
    }

    #[test]
    fn bless_preserves_unowned_files() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        fs::create_dir_all(&committed).unwrap();
        fs::write(committed.join("mod.rs"), b"hand-written").unwrap();
        fs::write(committed.join("old.txt"), b"old").unwrap();
        let ctx = GeneratorCtx::new(None);
        run_generator(
            &ctx,
            &committed,
            Mode::Bless,
            |out| -> std::result::Result<(), io::Error> {
                write_marker(&out.join("new.txt"), b"new");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(fs::read(committed.join("mod.rs")).unwrap(), b"hand-written");
        assert_eq!(fs::read(committed.join("old.txt")).unwrap(), b"old");
        assert!(committed.join("new.txt").exists());
    }

    #[test]
    fn generation_reformats_files() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        let file = committed.join("a.rs");
        write_marker(&file, b"foo");

        let ctx = GeneratorCtx::new(None);
        run_generator(
            &ctx,
            &committed,
            Mode::Bless,
            |out| -> std::result::Result<(), io::Error> {
                write_marker(&out.join("a.rs"), b"fn      main() {}");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(file).unwrap(),
            format!(
                r#"{GENERATED_MARKER}
fn main() {{}}
"#
            )
        );
    }
}
