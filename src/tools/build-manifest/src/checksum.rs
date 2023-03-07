use crate::manifest::{FileHash, Manifest};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

pub(crate) struct Checksums {
    cache_path: Option<PathBuf>,
    collected: Mutex<HashMap<PathBuf, String>>,
}

impl Checksums {
    pub(crate) fn new() -> Result<Self, Box<dyn Error>> {
        let cache_path = std::env::var_os("BUILD_MANIFEST_CHECKSUM_CACHE").map(PathBuf::from);

        let mut collected = HashMap::new();
        if let Some(path) = &cache_path {
            if path.is_file() {
                collected = serde_json::from_slice(&std::fs::read(path)?)?;
            }
        }

        Ok(Checksums { cache_path, collected: Mutex::new(collected) })
    }

    pub(crate) fn store_cache(&self) -> Result<(), Box<dyn Error>> {
        if let Some(path) = &self.cache_path {
            std::fs::write(path, &serde_json::to_vec(&self.collected)?)?;
        }
        Ok(())
    }

    pub(crate) fn fill_missing_checksums(&mut self, manifest: &mut Manifest) {
        let need_checksums = self.find_missing_checksums(manifest);
        if !need_checksums.is_empty() {
            self.collect_checksums(&need_checksums);
        }
        self.replace_checksums(manifest);
    }

    fn find_missing_checksums(&mut self, manifest: &mut Manifest) -> HashSet<PathBuf> {
        let collected = self.collected.lock().unwrap();
        let mut need_checksums = HashSet::new();
        crate::manifest::visit_file_hashes(manifest, |file_hash| {
            if let FileHash::Missing(path) = file_hash {
                let path = std::fs::canonicalize(path).unwrap();
                if !collected.contains_key(&path) {
                    need_checksums.insert(path);
                }
            }
        });
        need_checksums
    }

    fn replace_checksums(&mut self, manifest: &mut Manifest) {
        let collected = self.collected.lock().unwrap();
        crate::manifest::visit_file_hashes(manifest, |file_hash| {
            if let FileHash::Missing(path) = file_hash {
                let path = std::fs::canonicalize(path).unwrap();
                match collected.get(&path) {
                    Some(hash) => *file_hash = FileHash::Present(hash.clone()),
                    None => panic!("missing hash for file {}", path.display()),
                }
            }
        });
    }

    fn collect_checksums(&mut self, files: &HashSet<PathBuf>) {
        let collection_start = Instant::now();
        println!(
            "collecting hashes for {} tarballs across {} threads",
            files.len(),
            rayon::current_num_threads().min(files.len()),
        );

        files.par_iter().for_each(|path| match hash(path) {
            Ok(hash) => {
                self.collected.lock().unwrap().insert(path.clone(), hash);
            }
            Err(err) => eprintln!("error while fetching the hash for {}: {}", path.display(), err),
        });

        println!("collected {} hashes in {:.2?}", files.len(), collection_start.elapsed());
    }
}

fn hash(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut sha256 = Sha256::default();
    std::io::copy(&mut file, &mut sha256)?;
    Ok(hex::encode(sha256.finalize()))
}
