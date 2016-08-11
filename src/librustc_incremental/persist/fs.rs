// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! This module manages how the incremental compilation cache is represented in
//! the file system.
//!
//! Incremental compilation caches are managed according to a copy-on-write
//! strategy: Once a complete, consistent cache version is finalized, it is
//! never modified. Instead, when a subsequent compilation session is started,
//! the compiler will allocate a new version of the cache that starts out as
//! a copy of the previous version. Then only this new copy is modified and it
//! will not be visible to other processes until it is finalized. This ensures
//! that multiple compiler processes can be executed concurrently for the same
//! crate without interfering with each other or blocking each other.
//!
//! More concretely this is implemented via the following protocol:
//!
//! 1. For a newly started compilation session, the compiler allocates a
//!    new `session` directory within the incremental compilation directory.
//!    This session directory will have a unique name that ends with the suffix
//!    "-working" and that contains a creation timestamp.
//! 2. Next, the compiler looks for the newest finalized session directory,
//!    that is, a session directory from a previous compilation session that
//!    has been marked as valid and consistent. A session directory is
//!    considered finalized if the "-working" suffix in the directory name has
//!    been replaced by the SVH of the crate.
//! 3. Once the compiler has found a valid, finalized session directory, it will
//!    hard-link/copy its contents into the new "-working" directory. If all
//!    goes well, it will have its own, private copy of the source directory and
//!    subsequently not have to worry about synchronizing with other compiler
//!    processes.
//! 4. Now the compiler can do its normal compilation process, which involves
//!    reading and updating its private session directory.
//! 5. When compilation finishes without errors, the private session directory
//!    will be in a state where it can be used as input for other compilation
//!    sessions. That is, it will contain a dependency graph and cache artifacts
//!    that are consistent with the state of the source code it was compiled
//!    from, with no need to change them ever again. At this point, the compiler
//!    finalizes and "publishes" its private session directory by renaming it
//!    from "sess-{timestamp}-{random}-working" to "sess-{timestamp}-{SVH}".
//! 6. At this point the "old" session directory that we copied our data from
//!    at the beginning of the session has become obsolete because we have just
//!    published a more current version. Thus the compiler will delete it.
//!
//! ## Garbage Collection
//!
//! Naively following the above protocol might lead to old session directories
//! piling up if a compiler instance crashes for some reason before its able to
//! remove its private session directory. In order to avoid wasting disk space,
//! the compiler also does some garbage collection each time it is started in
//! incremental compilation mode. Specifically, it will scan the incremental
//! compilation directory for private session directories that are not in use
//! any more and will delete those. It will also delete any finalized session
//! directories for a given crate except for the most recent one.
//!
//! ## Synchronization
//!
//! There is some synchronization needed in order for the compiler to be able to
//! determine whether a given private session directory is not in used any more.
//! This is done by creating a lock file within each session directory and
//! locking it while the directory is still being used. Since file locks have
//! operating system support, we can rely on the lock being released if the
//! compiler process dies for some unexpected reason. Thus, when garbage
//! collecting private session directories, the collecting process can determine
//! whether the directory is still in use by trying to acquire a lock on the
//! file. If locking the file fails, the original process must still be alive.
//! If locking the file succeeds, we know that the owning process is not alive
//! any more and we can safely delete the directory.
//! There is still a small time window between the original process creating the
//! lock file and actually locking it. In order to minimize the chance that
//! another process tries to acquire the lock in just that instance, only
//! session directories that are older than a few seconds are considered for
//! garbage collection.
//!
//! Another case that has to be considered is what happens if one process
//! deletes a finalized session directory that another process is currently
//! trying to copy from. This case is also handled via the lock file. Before
//! a process starts copying a finalized session directory, it will acquire a
//! shared lock on the directory's lock file. Any garbage collecting process,
//! on the other hand, will acquire an exclusive lock on the lock file.
//! Thus, if a directory is being collected, any reader process will fail
//! acquiring the shared lock and will leave the directory alone. Conversely,
//! if a collecting process can't acquire the exclusive lock because the
//! directory is currently being read from, it will leave collecting that
//! directory to another process at a later point in time.
//! The exact same scheme is also used when reading the metadata hashes file
//! from an extern crate. When a crate is compiled, the hash values of its
//! metadata are stored in a file in its session directory. When the
//! compilation session of another crate imports the first crate's metadata,
//! it also has to read in the accompanying metadata hashes. It thus will access
//! the finalized session directory of all crates it links to and while doing
//! so, it will also place a read lock on that the respective session directory
//! so that it won't be deleted while the metadata hashes are loaded.
//!
//! ## Preconditions
//!
//! This system relies on two features being available in the file system in
//! order to work really well: file locking and hard linking.
//! If hard linking is not available (like on FAT) the data in the cache
//! actually has to be copied at the beginning of each session.
//! If file locking does not work reliably (like on NFS), some of the
//! synchronization will go haywire.
//! In both cases we recommend to locate the incremental compilation directory
//! on a file system that supports these things.
//! It might be a good idea though to try and detect whether we are on an
//! unsupported file system and emit a warning in that case. This is not yet
//! implemented.

use rustc::hir::svh::Svh;
use rustc::middle::cstore::LOCAL_CRATE;
use rustc::session::Session;
use rustc::ty::TyCtxt;
use rustc::util::fs as fs_util;
use rustc_data_structures::flock;
use rustc_data_structures::fnv::{FnvHashSet, FnvHashMap};

use std::ffi::OsString;
use std::fs as std_fs;
use std::io;
use std::mem;
use std::path::{Path, PathBuf};
use std::time::{UNIX_EPOCH, SystemTime, Duration};
use std::__rand::{thread_rng, Rng};
use syntax::ast;

const LOCK_FILE_NAME: &'static str = ".lock_file";
const DEP_GRAPH_FILENAME: &'static str = "dep-graph.bin";
const WORK_PRODUCTS_FILENAME: &'static str = "work-products.bin";
const METADATA_HASHES_FILENAME: &'static str = "metadata.bin";

pub fn dep_graph_path(sess: &Session) -> PathBuf {
    in_incr_comp_dir_sess(sess, DEP_GRAPH_FILENAME)
}

pub fn work_products_path(sess: &Session) -> PathBuf {
    in_incr_comp_dir_sess(sess, WORK_PRODUCTS_FILENAME)
}

pub fn metadata_hash_export_path(sess: &Session) -> PathBuf {
    in_incr_comp_dir_sess(sess, METADATA_HASHES_FILENAME)
}

pub fn metadata_hash_import_path(import_session_dir: &Path) -> PathBuf {
    import_session_dir.join(METADATA_HASHES_FILENAME)
}

pub fn lock_file_path(session_dir: &Path) -> PathBuf {
    session_dir.join(LOCK_FILE_NAME)
}

pub fn in_incr_comp_dir_sess(sess: &Session, file_name: &str) -> PathBuf {
    in_incr_comp_dir(&sess.incr_comp_session_dir(), file_name)
}

pub fn in_incr_comp_dir(incr_comp_session_dir: &Path, file_name: &str) -> PathBuf {
    incr_comp_session_dir.join(file_name)
}

/// Allocates the private session directory. The boolean in the Ok() result
/// indicates whether we should try loading a dep graph from the successfully
/// initialized directory, or not.
/// The post-condition of this fn is that we have a valid incremental
/// compilation session directory, if the result is `Ok`. A valid session
/// directory is one that contains a locked lock file. It may or may not contain
/// a dep-graph and work products from a previous session.
/// If the call fails, the fn may leave behind an invalid session directory.
/// The garbage collection will take care of it.
pub fn prepare_session_directory(tcx: TyCtxt) -> Result<bool, ()> {
    debug!("prepare_session_directory");

    // {incr-comp-dir}/{crate-name-and-disambiguator}
    let crate_dir = crate_path_tcx(tcx, LOCAL_CRATE);
    debug!("crate-dir: {}", crate_dir.display());

    let mut source_directories_already_tried = FnvHashSet();

    loop {
        // Allocate a session directory of the form:
        //
        // {incr-comp-dir}/{crate-name-and-disambiguator}/sess-{timestamp}-{random}-working
        //
        // If this fails, return an error, don't retry
        let session_dir = try!(alloc_session_dir(tcx.sess, &crate_dir));
        debug!("session-dir: {}", session_dir.display());

        // Lock the newly created session directory. If this fails, return an
        // error without retrying
        let directory_lock = try!(lock_directory(tcx.sess, &session_dir));

        let print_file_copy_stats = tcx.sess.opts.debugging_opts.incremental_info;

        // Find a suitable source directory to copy from. Ignore those that we
        // have already tried before.
        let source_directory = find_source_directory(&crate_dir,
                                                     &source_directories_already_tried);

        let source_directory = if let Some(dir) = source_directory {
            dir
        } else {
            // There's nowhere to copy from, we're done
            debug!("no source directory found. Continuing with empty session \
                    directory.");

            tcx.sess.init_incr_comp_session(session_dir, directory_lock);
            return Ok(false)
        };

        debug!("attempting to copy data from source: {}",
               source_directory.display());

        // Try copying over all files from the source directory
        if copy_files(&session_dir, &source_directory, print_file_copy_stats).is_ok() {
            debug!("successfully copied data from: {}",
                   source_directory.display());

            tcx.sess.init_incr_comp_session(session_dir, directory_lock);
            return Ok(true)
        } else {
             debug!("copying failed - trying next directory");

            // Something went wrong while trying to copy/link files from the
            // source directory. Try again with a different one.
            source_directories_already_tried.insert(source_directory);

            // Try to remove the session directory we just allocated. We don't
            // know if there's any garbage in it from the failed copy action.
            if let Err(err) = std_fs::remove_dir_all(&session_dir) {
                debug!("Failed to delete partly initialized session dir `{}`: {}",
                       session_dir.display(),
                       err);
            }
            mem::drop(directory_lock);
        }
    }
}

/// This function finalizes and thus 'publishes' the session directory by
/// renaming it to `sess-{timestamp}-{svh}` and releasing the file lock.
/// If there have been compilation errors, however, this function will just
/// delete the presumably invalid session directory.
pub fn finalize_session_directory(sess: &Session, svh: Svh) {
    if sess.opts.incremental.is_none() {
        return;
    }

    let incr_comp_session_dir: PathBuf = sess.incr_comp_session_dir().clone();

    if sess.has_errors() {
        // If there have been any errors during compilation, we don't want to
        // publish this session directory. Rather, we'll just delete it.

        debug!("finalize_session_directory() - invalidating session directory: {}",
                incr_comp_session_dir.display());

        if let Err(err) = std_fs::remove_dir_all(&*incr_comp_session_dir) {
            sess.warn(&format!("Error deleting incremental compilation \
                               session directory `{}`: {}",
                               incr_comp_session_dir.display(),
                               err));
        }
        sess.mark_incr_comp_session_as_invalid();
    }

    debug!("finalize_session_directory() - session directory: {}",
            incr_comp_session_dir.display());

    let old_sub_dir_name = incr_comp_session_dir.file_name()
                                                .unwrap()
                                                .to_string_lossy();
    assert_no_characters_lost(&old_sub_dir_name);

    // Keep the 'sess-{timestamp}' prefix, but replace the
    // '-{random-number}-working' part with the SVH of the crate
    let dash_indices: Vec<_> = old_sub_dir_name.match_indices("-")
                                               .map(|(idx, _)| idx)
                                               .collect();
    if dash_indices.len() != 3 {
        bug!("Encountered incremental compilation session directory with \
              malformed name: {}",
             incr_comp_session_dir.display())
    }

    // State: "sess-{timestamp}-"
    let mut new_sub_dir_name = String::from(&old_sub_dir_name[.. dash_indices[1] + 1]);

    // Append the svh
    new_sub_dir_name.push_str(&svh.to_string());

    // Create the full path
    let new_path = incr_comp_session_dir.parent().unwrap().join(new_sub_dir_name);
    debug!("finalize_session_directory() - new path: {}", new_path.display());

    match std_fs::rename(&*incr_comp_session_dir, &new_path) {
        Ok(_) => {
            debug!("finalize_session_directory() - directory renamed successfully");

            // This unlocks the directory
            sess.finalize_incr_comp_session(new_path);
        }
        Err(e) => {
            // Warn about the error. However, no need to abort compilation now.
            sess.warn(&format!("Error finalizing incremental compilation \
                               session directory `{}`: {}",
                               incr_comp_session_dir.display(),
                               e));

            debug!("finalize_session_directory() - error, marking as invalid");
            // Drop the file lock, so we can garage collect
            sess.mark_incr_comp_session_as_invalid();
        }
    }

    let _ = garbage_collect_session_directories(sess);
}

fn copy_files(target_dir: &Path,
              source_dir: &Path,
              print_stats_on_success: bool)
              -> Result<(), ()> {
    // We acquire a shared lock on the lock file of the directory, so that
    // nobody deletes it out from under us while we are reading from it.
    let lock_file_path = source_dir.join(LOCK_FILE_NAME);
    let _lock = if let Ok(lock) = flock::Lock::new(&lock_file_path,
                                                   false,   // don't wait,
                                                   false,   // don't create
                                                   false) { // not exclusive
        lock
    } else {
        // Could not acquire the lock, don't try to copy from here
        return Err(())
    };

    let source_dir_iterator = match source_dir.read_dir() {
        Ok(it) => it,
        Err(_) => return Err(())
    };

    let mut files_linked = 0;
    let mut files_copied = 0;

    for entry in source_dir_iterator {
        match entry {
            Ok(entry) => {
                let file_name = entry.file_name();

                if file_name.to_string_lossy() == LOCK_FILE_NAME {
                    continue;
                }

                let target_file_path = target_dir.join(file_name);
                let source_path = entry.path();

                debug!("copying into session dir: {}", source_path.display());
                match fs_util::link_or_copy(source_path, target_file_path) {
                    Ok(fs_util::LinkOrCopy::Link) => {
                        files_linked += 1
                    }
                    Ok(fs_util::LinkOrCopy::Copy) => {
                        files_copied += 1
                    }
                    Err(_) => return Err(())
                }
            }
            Err(_) => {
                return Err(())
            }
        }
    }

    if print_stats_on_success {
        println!("incr. comp. session directory: {} files hard-linked", files_linked);
        println!("incr. comp. session directory: {} files copied", files_copied);
    }

    Ok(())
}

/// Create a directory with a path of the form:
/// {crate_dir}/sess-{timestamp}-{random-number}-working
fn alloc_session_dir(sess: &Session,
                     crate_dir: &Path)
                     -> Result<PathBuf, ()> {
    let timestamp = timestamp_to_string(SystemTime::now());
    debug!("alloc_session_dir: timestamp = {}", timestamp);
    let random_number = thread_rng().next_u32();
    debug!("alloc_session_dir: random_number = {}", random_number);

    let directory_name = format!("sess-{}-{:x}-working", timestamp, random_number);
    debug!("alloc_session_dir: directory_name = {}", directory_name);
    let directory_path = crate_dir.join(directory_name);
    debug!("alloc_session_dir: directory_path = {}", directory_path.display());

    match fs_util::create_dir_racy(&directory_path) {
        Ok(()) => {
            debug!("alloc_session_dir: directory created successfully");
            Ok(directory_path)
        }
        Err(err) => {
            sess.err(&format!("incremental compilation: could not create \
                               session directory `{}`: {}",
                              directory_path.display(),
                              err));
            Err(())
        }
    }
}

/// Allocate a the lock-file and lock it.
fn lock_directory(sess: &Session,
                  session_dir: &Path)
                  -> Result<flock::Lock, ()> {
    let lock_file_path = session_dir.join(LOCK_FILE_NAME);
    debug!("lock_directory() - lock_file: {}", lock_file_path.display());

    match flock::Lock::new(&lock_file_path,
                           false, // don't wait
                           true,  // create the lock file
                           true) { // the lock should be exclusive
        Ok(lock) => Ok(lock),
        Err(err) => {
            sess.err(&format!("incremental compilation: could not create \
                               session directory lock file: {}", err));
            Err(())
        }
    }
}

/// Find the most recent published session directory that is not in the
/// ignore-list.
fn find_source_directory(crate_dir: &Path,
                         source_directories_already_tried: &FnvHashSet<PathBuf>)
                         -> Option<PathBuf> {
    let iter = crate_dir.read_dir()
                        .unwrap() // FIXME
                        .filter_map(|e| e.ok().map(|e| e.path()));

    find_source_directory_in_iter(iter, source_directories_already_tried)
}

fn find_source_directory_in_iter<I>(iter: I,
                                    source_directories_already_tried: &FnvHashSet<PathBuf>)
                                    -> Option<PathBuf>
    where I: Iterator<Item=PathBuf>
{
    let mut best_candidate = (UNIX_EPOCH, None);

    for session_dir in iter {
        if source_directories_already_tried.contains(&session_dir) ||
           !is_finalized(&session_dir.to_string_lossy()) {
            continue
        }

        let timestamp = {
            let directory_name = session_dir.file_name().unwrap().to_string_lossy();
            assert_no_characters_lost(&directory_name);

            extract_timestamp_from_session_dir(&directory_name)
                .unwrap_or_else(|_| {
                    bug!("unexpected incr-comp session dir: {}", session_dir.display())
                })
        };

        if timestamp > best_candidate.0 {
            best_candidate = (timestamp, Some(session_dir));
        }
    }

    best_candidate.1
}

fn is_finalized(directory_name: &str) -> bool {
    !directory_name.ends_with("-working")
}

fn is_session_directory(directory_name: &str) -> bool {
    directory_name.starts_with("sess-")
}

fn extract_timestamp_from_session_dir(directory_name: &str)
                                      -> Result<SystemTime, ()> {
    if !is_session_directory(directory_name) {
        return Err(())
    }

    let dash_indices: Vec<_> = directory_name.match_indices("-")
                                             .map(|(idx, _)| idx)
                                             .collect();
    if dash_indices.len() < 2 {
        return Err(())
    }

    string_to_timestamp(&directory_name[dash_indices[0]+1 .. dash_indices[1]])
}

fn timestamp_to_string(timestamp: SystemTime) -> String {
    let duration = timestamp.duration_since(UNIX_EPOCH).unwrap();
    let nanos = duration.as_secs() * 1_000_000_000 +
                (duration.subsec_nanos() as u64);
    format!("{:x}", nanos)
}

fn string_to_timestamp(s: &str) -> Result<SystemTime, ()> {
    let nanos_since_unix_epoch = u64::from_str_radix(s, 16);

    if nanos_since_unix_epoch.is_err() {
        return Err(())
    }

    let nanos_since_unix_epoch = nanos_since_unix_epoch.unwrap();

    let duration = Duration::new(nanos_since_unix_epoch / 1_000_000_000,
                                 (nanos_since_unix_epoch % 1_000_000_000) as u32);
    Ok(UNIX_EPOCH + duration)
}

fn crate_path_tcx(tcx: TyCtxt, cnum: ast::CrateNum) -> PathBuf {
    crate_path(tcx.sess, &tcx.crate_name(cnum), &tcx.crate_disambiguator(cnum))
}

/// Finds the session directory containing the correct metadata hashes file for
/// the given crate. In order to do that it has to compute the crate directory
/// of the given crate, and in there, look for the session directory with the
/// correct SVH in it.
/// Note that we have to match on the exact SVH here, not just the
/// crate's (name, disambiguator) pair. The metadata hashes are only valid for
/// the exact version of the binary we are reading from now (i.e. the hashes
/// are part of the dependency graph of a specific compilation session).
pub fn find_metadata_hashes_for(tcx: TyCtxt, cnum: ast::CrateNum) -> Option<PathBuf> {
    let crate_directory = crate_path_tcx(tcx, cnum);

    if !crate_directory.exists() {
        return None
    }

    let dir_entries = match crate_directory.read_dir() {
        Ok(dir_entries) => dir_entries,
        Err(e) => {
            tcx.sess
               .err(&format!("incremental compilation: Could not read crate directory `{}`: {}",
                             crate_directory.display(), e));
            return None
        }
    };

    let target_svh = tcx.sess.cstore.crate_hash(cnum).to_string();

    let sub_dir = find_metadata_hashes_iter(&target_svh, dir_entries.filter_map(|e| {
        e.ok().map(|e| e.file_name().to_string_lossy().into_owned())
    }));

    sub_dir.map(|sub_dir_name| crate_directory.join(&sub_dir_name))
}

fn find_metadata_hashes_iter<'a, I>(target_svh: &str, iter: I) -> Option<OsString>
    where I: Iterator<Item=String>
{
    for sub_dir_name in iter {
        if !is_session_directory(&sub_dir_name) || !is_finalized(&sub_dir_name) {
            // This is not a usable session directory
            continue
        }

        let is_match = if let Some(last_dash_pos) = sub_dir_name.rfind("-") {
            let candidate_svh = &sub_dir_name[last_dash_pos + 1 .. ];
            target_svh == candidate_svh
        } else {
            // some kind of invalid directory name
            continue
        };

        if is_match {
            return Some(OsString::from(sub_dir_name))
        }
    }

    None
}

fn crate_path(sess: &Session,
              crate_name: &str,
              crate_disambiguator: &str)
              -> PathBuf {
    use std::hash::{SipHasher, Hasher, Hash};

    let incr_dir = sess.opts.incremental.as_ref().unwrap().clone();

    // The full crate disambiguator is really long. A hash of it should be
    // sufficient.
    let mut hasher = SipHasher::new();
    crate_disambiguator.hash(&mut hasher);

    let crate_name = format!("{}-{:x}", crate_name, hasher.finish());
    incr_dir.join(crate_name)
}

fn assert_no_characters_lost(s: &str) {
    if s.contains('\u{FFFD}') {
        bug!("Could not losslessly convert '{}'.", s)
    }
}

pub fn garbage_collect_session_directories(sess: &Session) -> io::Result<()> {
    debug!("garbage_collect_session_directories() - begin");

    let session_directory = sess.incr_comp_session_dir();
    debug!("garbage_collect_session_directories() - session directory: {}",
        session_directory.display());

    let crate_directory = session_directory.parent().unwrap();
    debug!("garbage_collect_session_directories() - crate directory: {}",
        crate_directory.display());

    let mut deletion_candidates = vec![];
    let mut definitely_delete = vec![];

    for dir_entry in try!(crate_directory.read_dir()) {
        let dir_entry = match dir_entry {
            Ok(dir_entry) => dir_entry,
            _ => {
                // Ignore any errors
                continue
            }
        };

        let directory_name = dir_entry.file_name();
        let directory_name = directory_name.to_string_lossy();

        if !is_session_directory(&directory_name) {
            // This is something we don't know, leave it alone...
            continue
        }
        assert_no_characters_lost(&directory_name);

        if let Ok(file_type) = dir_entry.file_type() {
            if !file_type.is_dir() {
                // This is not a directory, skip it
                continue
            }
        } else {
            // Some error occurred while trying to determine the file type,
            // skip it
            continue
        }

        debug!("garbage_collect_session_directories() - inspecting: {}",
                directory_name);

        match extract_timestamp_from_session_dir(&directory_name) {
            Ok(timestamp) => {
                let lock_file_path = crate_directory.join(&*directory_name)
                                                    .join(LOCK_FILE_NAME);

                if !is_finalized(&directory_name) {
                    let ten_seconds = Duration::from_secs(10);

                    // When cleaning out "-working" session directories, i.e.
                    // session directories that might still be in use by another
                    // compiler instance, we only look a directories that are
                    // at least ten seconds old. This is supposed to reduce the
                    // chance of deleting a directory in the time window where
                    // the process has allocated the directory but has not yet
                    // acquired the file-lock on it.
                    if timestamp < SystemTime::now() - ten_seconds {
                        debug!("garbage_collect_session_directories() - \
                                attempting to collect");

                        // Try to acquire the directory lock. If we can't, it
                        // means that the owning process is still alive and we
                        // leave this directory alone.
                        match flock::Lock::new(&lock_file_path,
                                               false,  // don't wait
                                               false,  // don't create the lock-file
                                               true) { // get an exclusive lock
                            Ok(lock) => {
                                debug!("garbage_collect_session_directories() - \
                                        successfully acquired lock");

                                // Note that we are holding on to the lock
                                definitely_delete.push((dir_entry.path(),
                                                        Some(lock)));
                            }
                            Err(_) => {
                                debug!("garbage_collect_session_directories() - \
                                not collecting, still in use");
                            }
                        }
                    } else {
                        debug!("garbage_collect_session_directories() - \
                                private session directory too new");
                    }
                } else {
                    match flock::Lock::new(&lock_file_path,
                                           false,  // don't wait
                                           false,  // don't create the lock-file
                                           true) { // get an exclusive lock
                        Ok(lock) => {
                            debug!("garbage_collect_session_directories() - \
                                    successfully acquired lock");
                            debug!("garbage_collect_session_directories() - adding \
                                    deletion candidate: {}", directory_name);

                            // Note that we are holding on to the lock
                            deletion_candidates.push((timestamp,
                                                      dir_entry.path(),
                                                      Some(lock)));
                        }
                        Err(_) => {
                            debug!("garbage_collect_session_directories() - \
                            not collecting, still in use");
                        }
                    }
                }
            }
            Err(_) => {
                // Malformed timestamp in directory, delete it
                definitely_delete.push((dir_entry.path(), None));

                debug!("garbage_collect_session_directories() - encountered \
                        malformed session directory: {}", directory_name);
            }
        }
    }

    // Delete all but the most recent of the candidates
    for (path, lock) in all_except_most_recent(deletion_candidates) {
        debug!("garbage_collect_session_directories() - deleting `{}`",
                path.display());

        if let Err(err) = std_fs::remove_dir_all(&path) {
            sess.warn(&format!("Failed to garbage collect finalized incremental \
                                compilation session directory `{}`: {}",
                               path.display(),
                               err));
        }

        // Let's make it explicit that the file lock is released at this point,
        // or rather, that we held on to it until here
        mem::drop(lock);
    }

    for (path, lock) in definitely_delete {
        debug!("garbage_collect_session_directories() - deleting `{}`",
                path.display());

        if let Err(err) = std_fs::remove_dir_all(&path) {
            sess.warn(&format!("Failed to garbage collect incremental \
                                compilation session directory `{}`: {}",
                               path.display(),
                               err));
        }

        // Let's make it explicit that the file lock is released at this point,
        // or rather, that we held on to it until here
        mem::drop(lock);
    }

    Ok(())
}

fn all_except_most_recent(deletion_candidates: Vec<(SystemTime, PathBuf, Option<flock::Lock>)>)
                          -> FnvHashMap<PathBuf, Option<flock::Lock>> {
    let most_recent = deletion_candidates.iter()
                                         .map(|&(timestamp, _, _)| timestamp)
                                         .max();

    if let Some(most_recent) = most_recent {
        deletion_candidates.into_iter()
                           .filter(|&(timestamp, _, _)| timestamp != most_recent)
                           .map(|(_, path, lock)| (path, lock))
                           .collect()
    } else {
        FnvHashMap()
    }
}

#[test]
fn test_all_except_most_recent() {
    assert_eq!(all_except_most_recent(
        vec![
            (UNIX_EPOCH + Duration::new(4, 0), PathBuf::from("4"), None),
            (UNIX_EPOCH + Duration::new(1, 0), PathBuf::from("1"), None),
            (UNIX_EPOCH + Duration::new(5, 0), PathBuf::from("5"), None),
            (UNIX_EPOCH + Duration::new(3, 0), PathBuf::from("3"), None),
            (UNIX_EPOCH + Duration::new(2, 0), PathBuf::from("2"), None),
        ]).keys().cloned().collect::<FnvHashSet<PathBuf>>(),
        vec![
            PathBuf::from("1"),
            PathBuf::from("2"),
            PathBuf::from("3"),
            PathBuf::from("4"),
        ].into_iter().collect::<FnvHashSet<PathBuf>>()
    );

    assert_eq!(all_except_most_recent(
        vec![
        ]).keys().cloned().collect::<FnvHashSet<PathBuf>>(),
        FnvHashSet()
    );
}

#[test]
fn test_timestamp_serialization() {
    for i in 0 .. 1_000u64 {
        let time = UNIX_EPOCH + Duration::new(i * 3_434_578, (i as u32) * 239_676);
        let s = timestamp_to_string(time);
        assert_eq!(time, string_to_timestamp(&s).unwrap());
    }
}

#[test]
fn test_find_source_directory_in_iter() {
    let already_visited = FnvHashSet();

    // Find newest
    assert_eq!(find_source_directory_in_iter(
        vec![PathBuf::from("./sess-3234-0000"),
             PathBuf::from("./sess-2234-0000"),
             PathBuf::from("./sess-1234-0000")].into_iter(), &already_visited),
        Some(PathBuf::from("./sess-3234-0000")));

    // Filter out "-working"
    assert_eq!(find_source_directory_in_iter(
        vec![PathBuf::from("./sess-3234-0000-working"),
             PathBuf::from("./sess-2234-0000"),
             PathBuf::from("./sess-1234-0000")].into_iter(), &already_visited),
        Some(PathBuf::from("./sess-2234-0000")));

    // Handle empty
    assert_eq!(find_source_directory_in_iter(vec![].into_iter(), &already_visited),
               None);

    // Handle only working
    assert_eq!(find_source_directory_in_iter(
        vec![PathBuf::from("./sess-3234-0000-working"),
             PathBuf::from("./sess-2234-0000-working"),
             PathBuf::from("./sess-1234-0000-working")].into_iter(), &already_visited),
        None);
}

#[test]
fn test_find_metadata_hashes_iter()
{
    assert_eq!(find_metadata_hashes_iter("testsvh2",
        vec![
            String::from("sess-timestamp1-testsvh1"),
            String::from("sess-timestamp2-testsvh2"),
            String::from("sess-timestamp3-testsvh3"),
        ].into_iter()),
        Some(OsString::from("sess-timestamp2-testsvh2"))
    );

    assert_eq!(find_metadata_hashes_iter("testsvh2",
        vec![
            String::from("sess-timestamp1-testsvh1"),
            String::from("sess-timestamp2-testsvh2"),
            String::from("invalid-name"),
        ].into_iter()),
        Some(OsString::from("sess-timestamp2-testsvh2"))
    );

    assert_eq!(find_metadata_hashes_iter("testsvh2",
        vec![
            String::from("sess-timestamp1-testsvh1"),
            String::from("sess-timestamp2-testsvh2-working"),
            String::from("sess-timestamp3-testsvh3"),
        ].into_iter()),
        None
    );

    assert_eq!(find_metadata_hashes_iter("testsvh1",
        vec![
            String::from("sess-timestamp1-random1-working"),
            String::from("sess-timestamp2-random2-working"),
            String::from("sess-timestamp3-random3-working"),
        ].into_iter()),
        None
    );

    assert_eq!(find_metadata_hashes_iter("testsvh2",
        vec![
            String::from("timestamp1-testsvh2"),
            String::from("timestamp2-testsvh2"),
            String::from("timestamp3-testsvh2"),
        ].into_iter()),
        None
    );
}
