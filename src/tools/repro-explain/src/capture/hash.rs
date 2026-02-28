use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn parse_dep_info(path: &Path) -> Result<Vec<String>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read dep-info {}", path.display()))?;
    // Makefile continuation handling.
    let joined = raw.replace("\\\n", " ");
    let deps = joined
        .split_once(':')
        .map(|(_, rhs)| rhs)
        .unwrap_or("")
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .map(unescape_makefile_token)
        .collect::<Vec<_>>();
    Ok(deps)
}

fn unescape_makefile_token(tok: &str) -> String {
    tok.replace("\\ ", " ")
}

pub fn walk_files(root: &Path) -> Vec<std::path::PathBuf> {
    WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect()
}

pub fn sha256_files_parallel(
    paths: &[PathBuf],
) -> Result<std::collections::HashMap<PathBuf, String>> {
    let pairs = paths
        .par_iter()
        .map(|path| {
            let sha = sha256_file(path)?;
            Ok((path.clone(), sha))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(pairs.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::{sha256_file, sha256_files_parallel};

    #[test]
    fn computes_hashes_in_parallel_for_multiple_files() {
        let dir = tempdir().expect("tempdir");
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        fs::write(&a, "alpha").expect("write a");
        fs::write(&b, "beta").expect("write b");

        let map = sha256_files_parallel(&[a.clone(), b.clone()]).expect("parallel hash");
        assert_eq!(map.get(&a), Some(&sha256_file(&a).expect("hash a")));
        assert_eq!(map.get(&b), Some(&sha256_file(&b).expect("hash b")));
    }
}
