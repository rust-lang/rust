use std::fs::{read_link, symlink_metadata};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use tar::{Builder, Header, HeaderMode};
use walkdir::WalkDir;

use crate::compression::{CombinedEncoder, CompressionFormats, CompressionProfile};
use crate::util::*;

actor! {
    #[derive(Debug)]
    pub struct Tarballer {
        /// The input folder to be compressed.
        #[arg(value_name = "NAME")]
        input: String = "package",

        /// The prefix of the tarballs.
        #[arg(value_name = "PATH")]
        output: String = "./dist",

        /// The folder in which the input is to be found.
        #[arg(value_name = "DIR")]
        work_dir: String = "./workdir",

        /// The profile used to compress the tarball.
        #[arg(value_name = "FORMAT", default_value_t)]
        compression_profile: CompressionProfile,

        /// The formats used to compress the tarball.
        #[arg(value_name = "FORMAT", default_value_t)]
        compression_formats: CompressionFormats,

        /// Modification time that will be set for all files added to the archive.
        /// The default is the date of the first Rust commit from 2006.
        /// This serves for better reproducibility of the archives.
        #[arg(value_name = "FILE_MTIME", default_value_t = 1153704088)]
        override_file_mtime: u64,
    }
}

impl Tarballer {
    /// Generates the actual tarballs
    pub fn run(self) -> Result<()> {
        if let CompressionProfile::NoOp = self.compression_profile {
            return Ok(());
        }

        let tarball_name = self.output.clone() + ".tar";
        let encoder = CombinedEncoder::new(
            self.compression_formats
                .iter()
                .map(|f| f.encode(&tarball_name, self.compression_profile))
                .collect::<Result<Vec<_>>>()?,
        );

        // Sort files by their suffix, to group files with the same name from
        // different locations (likely identical) and files with the same
        // extension (likely containing similar data).
        // Sorting of file and directory paths also helps with the reproducibility
        // of the resulting archive.
        let (mut dirs, mut files) = get_recursive_paths(&self.work_dir, &self.input)
            .context("failed to collect file paths")?;
        dirs.sort();
        files.sort_by(|a, b| a.bytes().rev().cmp(b.bytes().rev()));

        // Write the tar into both encoded files. We write all directories
        // first, so files may be directly created. (See rust-lang/rustup.rs#1092.)
        let buf = BufWriter::with_capacity(1024 * 1024, encoder);
        let mut builder = Builder::new(buf);
        // Make uid, gid and mtime deterministic to improve reproducibility
        // The modification time of directories will be set to the date of the first Rust commit.
        // The modification time of files will be set to `override_file_mtime` (see `append_path`).
        builder.mode(HeaderMode::Deterministic);

        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(move || {
            for path in dirs {
                let src = Path::new(&self.work_dir).join(&path);
                builder
                    .append_dir(&path, &src)
                    .with_context(|| format!("failed to tar dir '{}'", src.display()))?;
            }
            for path in files {
                let src = Path::new(&self.work_dir).join(&path);
                append_path(&mut builder, &src, &path, self.override_file_mtime)
                    .with_context(|| format!("failed to tar file '{}'", src.display()))?;
            }
            builder
                .into_inner()
                .context("failed to finish writing .tar stream")?
                .into_inner()
                .ok()
                .unwrap()
                .finish()?;

            Ok(())
        })
    }
}

fn append_path<W: Write>(
    builder: &mut Builder<W>,
    src: &Path,
    path: &String,
    override_file_mtime: u64,
) -> Result<()> {
    let stat = symlink_metadata(src)?;
    let mut header = Header::new_gnu();
    header.set_metadata_in_mode(&stat, HeaderMode::Deterministic);
    header.set_mtime(override_file_mtime);

    if stat.file_type().is_symlink() {
        let link = read_link(src)?;
        builder.append_link(&mut header, path, &link)?;
    } else {
        if cfg!(windows) {
            // Windows doesn't really have a mode, so `tar` never marks files executable.
            // Use an extension whitelist to update files that usually should be so.
            const EXECUTABLES: [&str; 4] = ["exe", "dll", "py", "sh"];
            if let Some(ext) = src.extension().and_then(|s| s.to_str()) {
                if EXECUTABLES.contains(&ext) {
                    let mode = header.mode()?;
                    header.set_mode(mode | 0o111);
                }
            }
        }
        let file = open_file(src)?;
        builder.append_data(&mut header, path, &file)?;
    }
    Ok(())
}

/// Returns all `(directories, files)` under the source path.
fn get_recursive_paths<P, Q>(root: P, name: Q) -> Result<(Vec<String>, Vec<String>)>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let root = root.as_ref();
    let name = name.as_ref();

    if !name.is_relative() && !name.starts_with(root) {
        bail!("input '{}' is not in work dir '{}'", name.display(), root.display());
    }

    let mut dirs = vec![];
    let mut files = vec![];
    for entry in WalkDir::new(root.join(name)) {
        let entry = entry?;
        let path = entry.path().strip_prefix(root)?;
        let path = path_to_str(path)?;

        if entry.file_type().is_dir() {
            dirs.push(path.to_owned());
        } else {
            files.push(path.to_owned());
        }
    }
    Ok((dirs, files))
}
