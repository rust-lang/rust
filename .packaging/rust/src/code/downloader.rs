use super::utils::{self, run_and_printerror, Selection};
use crate::{Cli, Repo};

use curl::easy::Easy;
use flate2;
use flate2::bufread::GzDecoder;
use std::fs::{remove_dir_all, File, OpenOptions};
use std::io::prelude::*;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::Command;
use tar::Archive;

/// Download the given tarball to the specified location
fn download_tarball(repo_url: &str, download_filename: PathBuf) -> Result<(), String> {
    if std::path::Path::new(&download_filename).exists() {
        match std::fs::remove_file(&download_filename) {
            Ok(()) => {}
            Err(e) => {
                return Err(format!(
                    "unable to delete file, download location blocked: {}",
                    e
                ))
            }
        }
    }

    let mut file = match OpenOptions::new()
        .write(true)
        .read(true)
        .create(true)
        .open(&download_filename)
    {
        Ok(file_handler) => file_handler,
        Err(why) => panic!("couldn't create {}", why),
    };

    dbg!(&repo_url);
    let mut handle = Easy::new();
    handle.url(repo_url).unwrap();
    handle.follow_location(true).unwrap();

    let mut data = Vec::new();
    {
        let mut transfer = handle.transfer();
        transfer
            .write_function(|block_data| {
                data.extend_from_slice(block_data);
                Ok(block_data.len())
            })
            .unwrap();
        transfer.perform().unwrap();
    }

    dbg!(&download_filename);
    match file.write_all(&data) {
        Ok(_) => {}
        Err(e) => {
            return Err(format!(
                "Unable to write download {} to file {}. {}",
                repo_url,
                download_filename.display(),
                e
            ))
        }
    };
    file.sync_all().unwrap();
    Ok(())
}

fn unpack(tar_gz_file: &str, dst: &str) -> Result<(), std::io::Error> {
    let path = tar_gz_file;

    let tar_gz = File::open(path)?;
    let tar_gz_reader = BufReader::new(tar_gz);
    let tar = GzDecoder::new(tar_gz_reader);
    let mut archive = Archive::new(tar);
    archive.unpack(dst)?;

    Ok(())
}

fn download_repo(repo_url: &str, out_dir: PathBuf) -> Result<(), String> {
    if out_dir.exists() {
        // make space to download the latest head
        remove_dir_all(out_dir.clone()).expect("failed to delete existing directory!");
    }
    let mut command = Command::new("git");
    command.args(&["clone", "--depth", "1", repo_url, out_dir.to_str().unwrap()]);
    run_and_printerror(&mut command);
    Ok(())
}

fn download_single(repo: Repo, which: Selection) -> Result<(), String> {
    match repo {
        Repo::Local(_) => {}
        Repo::Stable => {
            let remote_tarball = utils::get_remote_tarball_url(which);
            let download_filename = utils::get_local_tarball_path(which);
            let mut download_checkfile = download_filename.clone();
            assert!(download_checkfile.set_extension("info"));
            if !download_checkfile.exists() {
                // Check to avoid repeated download
                download_tarball(&remote_tarball, download_filename.clone())?;
                if let Err(e) = std::fs::File::create(download_checkfile) {
                    return Err(e.to_string());
                };
            } else {
                println!(
                    "Skipping downloading {which} tarball since it already exists here: {:?}.",
                    download_filename
                );
            }

            let dest_dir = utils::get_enzyme_base_path();
            let unpack_checkfile =
                utils::get_local_repo_dir(repo, which).join("finished-unpacking.txt");
            if !unpack_checkfile.exists() {
                match unpack(
                    download_filename.to_str().unwrap(),
                    dest_dir.to_str().unwrap(),
                ) {
                    Ok(_) => {}
                    Err(e) => return Err(format!("failed unpacking: {e}")),
                };
                if let Err(e) = std::fs::File::create(unpack_checkfile) {
                    return Err(e.to_string());
                };
            } else {
                println!(
                    "Skipping unpacking {which} tarball since a checkfile already exists here: {:?}.",
                    unpack_checkfile
                );
            }
        }
        Repo::Head => {
            // TODO: avoid repeating based on commit
            let remote_path = utils::get_remote_repo_url(which);
            let dest_dir = utils::get_head_repo_dir(which);
            download_repo(&remote_path, dest_dir)?;
        }
    };
    Ok(())
}

/// This function can be used to download enzyme / rust from github.
///
/// Stable released are downloaded as tarballs and unpacked, the Head is taken from the github repo
/// directly. Data will be processed in `~/.cache/enzyme`.
/// Will not perform any action for those which are set to None or Some(Local(_)).
pub fn download(to_download: Cli) -> Result<(), String> {
    download_single(to_download.rust.clone(), Selection::Rust)?;
    download_single(to_download.enzyme, Selection::Enzyme)?;

    Ok(())
}
