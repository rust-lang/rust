use std::{
    fs::FileType,
    path::{Path, PathBuf},
};

use anyhow::Result;

use crate::not_bash::{fs2, rm_rf};

pub struct PreCacheCmd;

impl PreCacheCmd {
    /// Cleans the `./target` dir after the build such that only
    /// dependencies are cached on CI.
    pub fn run(self) -> Result<()> {
        let slow_tests_cookie = Path::new("./target/.slow_tests_cookie");
        if !slow_tests_cookie.exists() {
            panic!("slow tests were skipped on CI!")
        }
        rm_rf(slow_tests_cookie)?;

        for path in read_dir("./target/debug", FileType::is_file)? {
            // Can't delete yourself on windows :-(
            if !path.ends_with("xtask.exe") {
                rm_rf(&path)?
            }
        }

        fs2::remove_file("./target/.rustc_info.json")?;

        let to_delete = read_dir("./crates", FileType::is_dir)?
            .into_iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().replace('-', "_"))
            .collect::<Vec<_>>();

        for &dir in ["./target/debug/deps", "target/debug/.fingerprint"].iter() {
            for path in read_dir(dir, |_file_type| true)? {
                if path.ends_with("xtask.exe") {
                    continue;
                }
                let file_name = path.file_name().unwrap().to_string_lossy();
                let (stem, _) = match rsplit_once(&file_name, '-') {
                    Some(it) => it,
                    None => {
                        rm_rf(path)?;
                        continue;
                    }
                };
                let stem = stem.replace('-', "_");
                if to_delete.contains(&stem) {
                    rm_rf(path)?;
                }
            }
        }

        Ok(())
    }
}
fn read_dir(path: impl AsRef<Path>, cond: impl Fn(&FileType) -> bool) -> Result<Vec<PathBuf>> {
    read_dir_impl(path.as_ref(), &cond)
}

fn read_dir_impl(path: &Path, cond: &dyn Fn(&FileType) -> bool) -> Result<Vec<PathBuf>> {
    let mut res = Vec::new();
    for entry in path.read_dir()? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if cond(&file_type) {
            res.push(entry.path())
        }
    }
    Ok(res)
}

fn rsplit_once(haystack: &str, delim: char) -> Option<(&str, &str)> {
    let mut split = haystack.rsplitn(2, delim);
    let suffix = split.next()?;
    let prefix = split.next()?;
    Some((prefix, suffix))
}
