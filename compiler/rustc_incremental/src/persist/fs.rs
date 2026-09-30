//! This module manages how the incremental compilation cache is represented in
//! the file system.
//!
//! Incremental compilation caches are managed according to a rebuild from
//! scratch strategy: Once a complete, consistent cache version is finalized, it
//! is never modified. Instead, when a subsequent compilation session is started,
//! the compiler will allocate a new version of the cache that starts out empty.
//! Then only this new directory is written to and it will not be visible to
//! other processes until it is finalized. This ensures that multiple compiler
//! processes can be executed concurrently for the same crate without
//! interfering with each other or blocking each other.
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
//!    obtain a shared lock on the directory. If this succeeds, it will have
//!    read-only access to the old session directory without having to worry
//!    about synchronizing with other compiler processes.
//! 4. Now the compiler can do its normal compilation process, which involves
//!    writing to its private session directory. Possibly by hardlinking
//!    existing files from the old session directory if they haven't changed.
//! 5. When compilation finishes without errors, the private session directory
//!    will be in a state where it can be used as input for other compilation
//!    sessions. That is, it will contain a dependency graph and cache artifacts
//!    that are consistent with the state of the source code it was compiled
//!    from, with no need to change them ever again. At this point, the compiler
//!    finalizes and "publishes" its private session directory by renaming it
//!    from "s-{timestamp}-{random}-working" to "s-{timestamp}-{SVH}".
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
//! determine whether a given private session directory is not in use any more.
//! This is done by creating a lock file for each session directory and
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
//! reading from. This case is also handled via the lock file. Before a process
//! starts reading from a finalized session directory, it will acquire a shared
//! lock on the directory's lock file. Any garbage collecting process, on the
//! other hand, will acquire an exclusive lock on the lock file. Thus, if a
//! directory is being collected, any reader process will fail acquiring the
//! shared lock and will leave the directory alone. Conversely, if a collecting
//! process can't acquire the exclusive lock because the directory is currently
//! being read from, it will leave collecting that directory to another process
//! at a later point in time.
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

use std::fs as std_fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rand::{RngCore, rng};
use rustc_data_structures::base_n::{BaseNString, CASE_INSENSITIVE, ToBaseN};
use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_data_structures::{base_n, flock};
use rustc_fs_util::try_canonicalize;
use rustc_middle::dep_graph::WorkProduct;
use rustc_session::config::OutputType;
use rustc_session::{IncrCompSession, Session, StableCrateId};
use rustc_span::{Symbol, bug};
use tracing::debug;

use crate::diagnostics;

#[cfg(test)]
mod tests;

const LOCK_FILE_EXT: &str = ".lock";
const DEP_GRAPH_FILENAME: &str = "dep-graph.bin";
const STAGING_DEP_GRAPH_FILENAME: &str = "dep-graph.part.bin";
const WORK_PRODUCTS_FILENAME: &str = "work-products.bin";
const QUERY_CACHE_FILENAME: &str = "query-cache.bin";

// We encode integers using the following base, so they are shorter than decimal
// or hexadecimal numbers (we want short file and directory names). Since these
// numbers will be used in file names, we choose an encoding that is not
// case-sensitive (as opposed to base64, for example).
const INT_ENCODE_BASE: usize = base_n::CASE_INSENSITIVE;

/// Returns the path to a previous session's dependency graph.
pub(crate) fn old_dep_graph_path(incr_comp_session: &IncrCompSession) -> Option<PathBuf> {
    in_old_incr_comp_dir_sess(incr_comp_session, DEP_GRAPH_FILENAME)
}

/// Returns the path to a session's dependency graph.
pub(crate) fn dep_graph_path(incr_comp_session: &IncrCompSession) -> PathBuf {
    in_incr_comp_dir_sess(incr_comp_session, DEP_GRAPH_FILENAME)
}

/// Returns the path to a session's staging dependency graph.
///
/// On the difference between dep-graph and staging dep-graph,
/// see `build_dep_graph`.
pub(crate) fn staging_dep_graph_path(incr_comp_session: &IncrCompSession) -> PathBuf {
    in_incr_comp_dir_sess(incr_comp_session, STAGING_DEP_GRAPH_FILENAME)
}

pub(crate) fn old_work_products_path(incr_comp_session: &IncrCompSession) -> Option<PathBuf> {
    in_old_incr_comp_dir_sess(incr_comp_session, WORK_PRODUCTS_FILENAME)
}

pub(crate) fn work_products_path(incr_comp_session: &IncrCompSession) -> PathBuf {
    in_incr_comp_dir_sess(incr_comp_session, WORK_PRODUCTS_FILENAME)
}

/// Returns the path to a previous session's query cache.
pub(crate) fn old_query_cache_path(incr_comp_session: &IncrCompSession) -> Option<PathBuf> {
    in_old_incr_comp_dir_sess(incr_comp_session, QUERY_CACHE_FILENAME)
}

/// Returns the path to a session's query cache.
pub(crate) fn query_cache_path(incr_comp_session: &IncrCompSession) -> PathBuf {
    in_incr_comp_dir_sess(incr_comp_session, QUERY_CACHE_FILENAME)
}

/// Locks a given session directory.
fn lock_file_path(session_dir: &Path) -> PathBuf {
    let crate_dir = session_dir.parent().unwrap();

    let directory_name = session_dir
        .file_name()
        .unwrap()
        .to_str()
        .expect("malformed session dir name: contains non-Unicode characters");

    let dash_indices: Vec<_> = directory_name.match_indices('-').map(|(idx, _)| idx).collect();
    if dash_indices.len() != 3 {
        bug!(
            "Encountered incremental compilation session directory with \
              malformed name: {}",
            session_dir.display()
        )
    }

    crate_dir.join(&directory_name[0..dash_indices[2]]).with_extension(&LOCK_FILE_EXT[1..])
}

/// Returns the path for a given filename within the incremental compilation directory
/// in the previous session.
pub fn in_old_incr_comp_dir_sess(
    incr_comp_session: &IncrCompSession,
    file_name: &str,
) -> Option<PathBuf> {
    incr_comp_session.old_session_directory.as_ref().map(|dir| dir.join(file_name))
}

/// Returns the path for a given filename within the incremental compilation directory
/// in the current session.
pub fn in_incr_comp_dir_sess(incr_comp_session: &IncrCompSession, file_name: &str) -> PathBuf {
    incr_comp_session.new_session_directory.join(file_name)
}

/// Allocates the private session directory.
///
/// If the result of this function is `Ok`, we have a valid incremental
/// compilation session directory. A valid session
/// directory is one that contains a locked lock file. It may or may not contain
/// a dep-graph and work products from a previous session.
///
/// This always attempts to load a dep-graph from the directory.
/// If loading fails for some reason, we fallback to a disabled `DepGraph`.
/// See [`rustc_interface::queries::dep_graph`].
///
/// If this function returns an error, it may leave behind an invalid session directory.
/// The garbage collection will take care of it.
///
/// [`rustc_interface::queries::dep_graph`]: ../../rustc_interface/struct.Queries.html#structfield.dep_graph
pub(crate) fn prepare_session_directory(
    sess: &Session,
    crate_name: Symbol,
    stable_crate_id: StableCrateId,
) -> IncrCompSession {
    assert!(sess.opts.incremental.is_some());

    let _timer = sess.timer("incr_comp_prepare_session_directory");

    debug!("prepare_session_directory");

    // {incr-comp-dir}/{crate-name-and-disambiguator}
    let crate_dir = crate_path(sess, crate_name, stable_crate_id);
    debug!("crate-dir: {}", crate_dir.display());
    create_dir(sess, &crate_dir, "crate");

    // Hack: canonicalize the path *after creating the directory*
    // because, on windows, long paths can cause problems;
    // canonicalization inserts this weird prefix that makes windows
    // tolerate long paths.
    let crate_dir = match try_canonicalize(&crate_dir) {
        Ok(v) => v,
        Err(err) => {
            sess.dcx().emit_fatal(diagnostics::CanonicalizePath { path: crate_dir, err });
        }
    };

    // Generate a session directory of the form:
    //
    // {incr-comp-dir}/{crate-name-and-disambiguator}/s-{timestamp}-{random}-working
    let new_session_dir = generate_session_dir_path(&crate_dir);
    debug!("session-dir: {}", new_session_dir.display());

    // Lock the new session directory. If this fails, return an
    // error without retrying
    let new_session_directory = lock_directory(sess, &new_session_dir, true /* new_session */)
        .expect("should emit fatal error on lock fail");

    // Find a suitable source directory to copy from. Ignore those that we
    // have already tried before.
    let old_source_directory = find_source_directory(sess, &crate_dir);

    let old_session_directory = if let Some(old_source_directory) = old_source_directory {
        debug!("attempting to use: {}", old_source_directory.display());
        Some(old_source_directory)
    } else {
        debug!("no source directory found. Continuing with empty session directory.");
        None
    };

    IncrCompSession { old_session_directory, new_session_directory }
}

/// This function finalizes and thus 'publishes' the session directory by
/// renaming it to `s-{timestamp}-{svh}` and releasing the file lock.
/// This must not be called if there have been any compilation errors.
pub fn finalize_session_directory(
    sess: &Session,
    incr_comp_session: Option<IncrCompSession>,
    svh: Option<Svh>,
) {
    assert!(sess.dcx().has_errors_or_delayed_bugs().is_none());

    if sess.opts.incremental.is_none() {
        return;
    }
    let mut incr_comp_session = incr_comp_session.unwrap();
    // The svh is always produced when incr. comp. is enabled.
    let svh = svh.unwrap();

    let _timer = sess.timer("incr_comp_finalize_session_directory");

    let incr_comp_session_dir = &*incr_comp_session.new_session_directory;

    debug!("finalize_session_directory() - session directory: {}", incr_comp_session_dir.display());

    let mut sub_dir_name = incr_comp_session_dir
        .file_name()
        .unwrap()
        .to_str()
        .expect("malformed session dir name: contains non-Unicode characters")
        .to_string();

    // Keep the 's-{timestamp}-{random-number}' prefix, but replace "working" with the SVH of the crate
    sub_dir_name.truncate(sub_dir_name.len() - "working".len());
    // Double-check that we kept this: "s-{timestamp}-{random-number}-"
    assert!(sub_dir_name.ends_with('-'), "{:?}", sub_dir_name);
    assert!(sub_dir_name.as_bytes().iter().filter(|b| **b == b'-').count() == 3);

    // Append the SVH
    sub_dir_name.push_str(&svh.as_u128().to_base_fixed_len(CASE_INSENSITIVE));

    // Create the full path
    let new_path = incr_comp_session_dir.parent().unwrap().join(&*sub_dir_name);
    debug!("finalize_session_directory() - new path: {}", new_path.display());

    let result = std_fs::rename(incr_comp_session_dir, &new_path).or_else(|e| {
        if !cfg!(windows) || e.kind() != ErrorKind::PermissionDenied {
            return Err(e);
        }

        // On ReFS, renaming a directory that contains a hard link to the metadata workproduct file
        // can fail if it is being used by another process (such as another rustc instance).
        // As a fallback, we try to replace the hard link with a copy, which should allow the
        // rename to succeed.
        // See https://github.com/rust-lang/rust/issues/151181
        if let Err(err) = replace_hard_link_with_copy(&in_incr_comp_dir_sess(
            &incr_comp_session,
            &format!(
                "{}.{}",
                WorkProduct::METADATA_WORKPRODUCT_CGU_NAME,
                OutputType::Metadata.extension()
            ),
        )) {
            debug!("finalize_session_directory() - error replacing hard link with copy: {}", err);
        }

        rename_path_with_retry(incr_comp_session_dir, &new_path, 3)
    });

    match result {
        Ok(_) => {
            debug!("finalize_session_directory() - directory renamed successfully");
        }
        Err(e) => {
            // Warn about the error. However, no need to abort compilation now.
            sess.dcx().emit_note(diagnostics::Finalize { path: incr_comp_session_dir, err: e });

            debug!("finalize_session_directory() - error");
        }
    }

    // Unlock the old session directory now that we will no longer read from it.
    incr_comp_session.old_session_directory = None;

    let _ = garbage_collect_session_directories(sess, &incr_comp_session);
}

pub(crate) fn invalidate_old_session_dir(sess: &Session, incr_comp_session: &mut IncrCompSession) {
    if let Some(old_incr_comp_session_dir) = incr_comp_session.old_session_directory.take() {
        let res = try {
            let sess_dir_iterator = old_incr_comp_session_dir.read_dir()?;
            for entry in sess_dir_iterator {
                let entry = entry?;
                safe_remove_file(&entry.path())?
            }
        };
        if let Err(err) = res {
            sess.dcx().emit_err(diagnostics::DeleteIncompatible {
                path: (*old_incr_comp_session_dir).to_owned(),
                err,
            });
        }
    }
}

/// Generates unique directory path of the form:
/// {crate_dir}/s-{timestamp}-{random-number}-working
fn generate_session_dir_path(crate_dir: &Path) -> PathBuf {
    let timestamp = timestamp_to_string(SystemTime::now());
    debug!("generate_session_dir_path: timestamp = {}", timestamp);
    let random_number = rng().next_u32();
    debug!("generate_session_dir_path: random_number = {}", random_number);

    // Chop the first 3 characters off the timestamp. Those 3 bytes will be zero for a while.
    let (zeroes, timestamp) = timestamp.split_at(3);
    assert_eq!(zeroes, "000");
    let directory_name =
        format!("s-{}-{}-working", timestamp, random_number.to_base_fixed_len(CASE_INSENSITIVE));
    debug!("generate_session_dir_path: directory_name = {}", directory_name);
    let directory_path = crate_dir.join(directory_name);
    debug!("generate_session_dir_path: directory_path = {}", directory_path.display());
    directory_path
}

fn create_dir(sess: &Session, path: &Path, dir_tag: &str) {
    match std_fs::create_dir_all(path) {
        Ok(()) => {
            debug!("{} directory created successfully", dir_tag);
        }
        Err(err) => {
            sess.dcx().emit_fatal(diagnostics::CreateIncrCompDir { tag: dir_tag, path, err })
        }
    }
}

/// Allocate the lock-file, lock it and create the session directory if requested.
fn lock_directory(
    sess: &Session,
    session_dir: &Path,
    new_session: bool,
) -> Option<flock::LockedDir> {
    let lock_file_path = lock_file_path(session_dir);
    debug!("lock_directory() - lock_file: {}", lock_file_path.display());

    match flock::LockedDir::try_lock(
        session_dir.to_owned(),
        &lock_file_path,
        new_session, // create
        new_session, // exclusive
    ) {
        Ok(lock) => {
            // Now that we have the lock, we can actually create the session
            // directory
            if new_session {
                create_dir(sess, &session_dir, "session");
            }

            Some(lock)
        }
        Err(lock_err) => {
            let is_unsupported_lock = flock::Lock::error_unsupported(&lock_err);
            let diag = diagnostics::CreateLock {
                lock_err,
                session_dir,
                is_unsupported_lock,
                is_cargo: rustc_session::utils::was_invoked_from_cargo(),
            };
            if new_session {
                sess.dcx().emit_fatal(diag);
            } else {
                sess.dcx().emit_warn(diag);
                None
            }
        }
    }
}

fn delete_session_dir_lock_file(sess: &Session, lock_file_path: &Path) {
    if let Err(err) = safe_remove_file(lock_file_path) {
        sess.dcx().emit_warn(diagnostics::DeleteLock { path: lock_file_path, err });
    }
}

/// Finds the most recent published session directory.
fn find_source_directory(sess: &Session, crate_dir: &Path) -> Option<flock::LockedDir> {
    let iter = crate_dir
        .read_dir()
        .unwrap() // FIXME
        .filter_map(|e| e.ok().map(|e| e.path()));

    find_source_directory_in_iter(iter)
        .and_then(|session_dir| lock_directory(sess, &session_dir, false /* new_session */))
}

fn find_source_directory_in_iter<I>(iter: I) -> Option<PathBuf>
where
    I: Iterator<Item = PathBuf>,
{
    let mut best_candidate = (UNIX_EPOCH, None);

    for session_dir in iter {
        debug!("find_source_directory_in_iter - inspecting `{}`", session_dir.display());

        let Some(directory_name) = session_dir.file_name().unwrap().to_str() else {
            debug!("find_source_directory_in_iter - ignoring");
            continue;
        };

        if !is_session_directory(&directory_name) || !is_finalized(&directory_name) {
            debug!("find_source_directory_in_iter - ignoring");
            continue;
        }

        let timestamp = match extract_timestamp_from_session_dir(&directory_name) {
            Ok(timestamp) => timestamp,
            Err(e) => {
                debug!("unexpected incr-comp session dir: {}: {}", session_dir.display(), e);
                continue;
            }
        };

        if timestamp > best_candidate.0 {
            best_candidate = (timestamp, Some(session_dir.clone()));
        }
    }

    best_candidate.1
}

fn is_finalized(directory_name: &str) -> bool {
    !directory_name.ends_with("-working")
}

fn is_session_directory(directory_name: &str) -> bool {
    directory_name.starts_with("s-") && !directory_name.ends_with(LOCK_FILE_EXT)
}

fn is_session_directory_lock_file(file_name: &str) -> bool {
    file_name.starts_with("s-") && file_name.ends_with(LOCK_FILE_EXT)
}

fn extract_timestamp_from_session_dir(directory_name: &str) -> Result<SystemTime, &'static str> {
    if !is_session_directory(directory_name) {
        return Err("not a directory");
    }

    let dash_indices: Vec<_> = directory_name.match_indices('-').map(|(idx, _)| idx).collect();
    if dash_indices.len() != 3 {
        return Err("not three dashes in name");
    }

    string_to_timestamp(&directory_name[dash_indices[0] + 1..dash_indices[1]])
}

fn timestamp_to_string(timestamp: SystemTime) -> BaseNString {
    let duration = timestamp.duration_since(UNIX_EPOCH).unwrap();
    let micros: u64 = duration.as_micros().try_into().unwrap();
    micros.to_base_fixed_len(CASE_INSENSITIVE)
}

fn string_to_timestamp(s: &str) -> Result<SystemTime, &'static str> {
    let micros_since_unix_epoch = match u64::from_str_radix(s, INT_ENCODE_BASE as u32) {
        Ok(micros) => micros,
        Err(_) => return Err("timestamp not an int"),
    };

    let duration = Duration::from_micros(micros_since_unix_epoch);
    Ok(UNIX_EPOCH + duration)
}

fn crate_path(sess: &Session, crate_name: Symbol, stable_crate_id: StableCrateId) -> PathBuf {
    let incr_dir = sess.opts.incremental.as_ref().unwrap().clone();

    let crate_name =
        format!("{crate_name}-{}", stable_crate_id.as_u64().to_base_fixed_len(CASE_INSENSITIVE));
    incr_dir.join(crate_name)
}

fn is_old_enough_to_be_collected(timestamp: SystemTime) -> bool {
    timestamp < SystemTime::now() - Duration::from_secs(10)
}

/// Runs garbage collection for the current session.
pub(crate) fn garbage_collect_session_directories(
    sess: &Session,
    incr_comp_session: &IncrCompSession,
) -> io::Result<()> {
    debug!("garbage_collect_session_directories() - begin");

    let session_directory = &*incr_comp_session.new_session_directory;

    debug!(
        "garbage_collect_session_directories() - session directory: {}",
        session_directory.display()
    );

    let crate_directory = session_directory.parent().unwrap();
    debug!(
        "garbage_collect_session_directories() - crate directory: {}",
        crate_directory.display()
    );

    // First do a pass over the crate directory, collecting lock files and
    // session directories
    let mut session_directories = FxIndexSet::default();
    let mut lock_files = UnordSet::default();

    for dir_entry in crate_directory.read_dir()? {
        let Ok(dir_entry) = dir_entry else {
            // Ignore any errors
            continue;
        };

        let entry_name = dir_entry.file_name();
        let Some(entry_name) = entry_name.to_str() else {
            continue;
        };

        if is_session_directory_lock_file(&entry_name) {
            lock_files.insert(entry_name.to_string());
        } else if is_session_directory(&entry_name) {
            session_directories.insert(entry_name.to_string());
        } else {
            // This is something we don't know, leave it alone
        }
    }
    session_directories.sort();

    // Now map from lock files to session directories
    let lock_file_to_session_dir: UnordMap<String, Option<String>> = lock_files
        .into_items()
        .map(|lock_file_name| {
            assert!(lock_file_name.ends_with(LOCK_FILE_EXT));
            let dir_prefix_end = lock_file_name.len() - LOCK_FILE_EXT.len();
            let session_dir = {
                let dir_prefix = &lock_file_name[0..dir_prefix_end];
                session_directories.iter().find(|dir_name| dir_name.starts_with(dir_prefix))
            };
            (lock_file_name, session_dir.map(String::clone))
        })
        .into();

    // Delete all lock files, that don't have an associated directory. They must
    // be some kind of leftover
    for (lock_file_name, directory_name) in
        lock_file_to_session_dir.items().into_sorted_stable_ord()
    {
        if directory_name.is_none() {
            let Ok(timestamp) = extract_timestamp_from_session_dir(lock_file_name) else {
                debug!(
                    "found lock-file with malformed timestamp: {}",
                    crate_directory.join(&lock_file_name).display()
                );
                // Ignore it
                continue;
            };

            let lock_file_path = crate_directory.join(&*lock_file_name);

            if is_old_enough_to_be_collected(timestamp) {
                debug!(
                    "garbage_collect_session_directories() - deleting \
                    garbage lock file: {}",
                    lock_file_path.display()
                );
                delete_session_dir_lock_file(sess, &lock_file_path);
            } else {
                debug!(
                    "garbage_collect_session_directories() - lock file with \
                    no session dir not old enough to be collected: {}",
                    lock_file_path.display()
                );
            }
        }
    }

    // Filter out `None` directories
    let lock_file_to_session_dir: UnordMap<String, String> = lock_file_to_session_dir
        .into_items()
        .filter_map(|(lock_file_name, directory_name)| directory_name.map(|n| (lock_file_name, n)))
        .into();

    // Delete all session directories that don't have a lock file.
    for directory_name in session_directories {
        if !lock_file_to_session_dir.items().any(|(_, dir)| *dir == directory_name) {
            let path = crate_directory.join(directory_name);
            if let Err(err) = std_fs::remove_dir_all(&path) {
                sess.dcx().emit_warn(diagnostics::InvalidGcFailed { path: &path, err });
            }
        }
    }

    // Now garbage collect the valid session directories.
    let deletion_candidates =
        lock_file_to_session_dir.items().filter_map(|(lock_file_name, directory_name)| {
            debug!("garbage_collect_session_directories() - inspecting: {}", directory_name);

            let Ok(timestamp) = extract_timestamp_from_session_dir(directory_name) else {
                debug!(
                    "found session-dir with malformed timestamp: {}",
                    crate_directory.join(directory_name).display()
                );
                // Ignore it
                return None;
            };

            if is_finalized(directory_name) {
                let lock_file_path = crate_directory.join(lock_file_name);
                match flock::Lock::try_lock(
                    &lock_file_path,
                    false, // don't create the lock-file
                    true,
                ) {
                    // get an exclusive lock
                    Ok(lock) => {
                        debug!(
                            "garbage_collect_session_directories() - \
                            successfully acquired lock"
                        );
                        debug!(
                            "garbage_collect_session_directories() - adding \
                            deletion candidate: {}",
                            directory_name
                        );

                        // Note that we are holding on to the lock
                        return Some((crate_directory.join(directory_name), lock));
                    }
                    Err(_) => {
                        debug!(
                            "garbage_collect_session_directories() - \
                            not collecting, still in use"
                        );
                    }
                }
            } else if is_old_enough_to_be_collected(timestamp) {
                // When cleaning out "-working" session directories, i.e.
                // session directories that might still be in use by another
                // compiler instance, we only look a directories that are
                // at least ten seconds old. This is supposed to reduce the
                // chance of deleting a directory in the time window where
                // the process has allocated the directory but has not yet
                // acquired the file-lock on it.

                // Try to acquire the directory lock. If we can't, it
                // means that the owning process is still alive and we
                // leave this directory alone.
                let lock_file_path = crate_directory.join(lock_file_name);
                match flock::Lock::try_lock(
                    &lock_file_path,
                    false, // don't create the lock-file
                    true,
                ) {
                    // get an exclusive lock
                    Ok(lock) => {
                        debug!(
                            "garbage_collect_session_directories() - \
                            successfully acquired lock"
                        );

                        delete_old(sess, &crate_directory.join(directory_name));

                        // Let's make it explicit that the file lock is released at this point,
                        // or rather, that we held on to it until here
                        drop(lock);
                    }
                    Err(_) => {
                        debug!(
                            "garbage_collect_session_directories() - \
                            not collecting, still in use"
                        );
                    }
                }
            } else {
                debug!(
                    "garbage_collect_session_directories() - not finalized, not \
                    old enough"
                );
            }
            None
        });

    // Delete all but the most recent of the candidates
    deletion_candidates.all(|(path, lock)| {
        debug!("garbage_collect_session_directories() - deleting `{}`", path.display());

        if let Err(err) = std_fs::remove_dir_all(&path) {
            sess.dcx().emit_warn(diagnostics::FinalizedGcFailed { path: &path, err });
        } else {
            delete_session_dir_lock_file(sess, &lock_file_path(&path));
        }

        // Let's make it explicit that the file lock is released at this point,
        // or rather, that we held on to it until here
        drop(lock);
        true
    });

    Ok(())
}

fn delete_old(sess: &Session, path: &Path) {
    debug!("garbage_collect_session_directories() - deleting `{}`", path.display());

    if let Err(err) = std_fs::remove_dir_all(path) {
        sess.dcx().emit_warn(diagnostics::SessionGcFailed { path, err });
    } else {
        delete_session_dir_lock_file(sess, &lock_file_path(path));
    }
}

fn safe_remove_file(p: &Path) -> io::Result<()> {
    match std_fs::remove_file(p) {
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        result => result,
    }
}

// On Windows the compiler would sometimes fail to rename the session directory because
// the OS thought something was still being accessed in it. So we retry a few times to give
// the OS time to catch up.
// See https://github.com/rust-lang/rust/issues/86929.
fn rename_path_with_retry(from: &Path, to: &Path, mut retries_left: usize) -> std::io::Result<()> {
    loop {
        match std_fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) => {
                if retries_left > 0 && e.kind() == ErrorKind::PermissionDenied {
                    // Try again after a short waiting period.
                    std::thread::sleep(Duration::from_millis(50));
                    retries_left -= 1;
                } else {
                    return Err(e);
                }
            }
        }
    }
}

/// Turns a hard link of the file at `path` into a copy.
fn replace_hard_link_with_copy(path: &Path) -> std::io::Result<()> {
    let tmp_name = path.with_added_extension("tmp");

    // In case a stale temporary file was linked from a previous failed attempt.
    safe_remove_file(&tmp_name)?;

    std_fs::copy(path, &tmp_name).and_then(|_| std_fs::rename(&tmp_name, path)).inspect_err(|_| {
        let _ = safe_remove_file(&tmp_name);
    })
}
