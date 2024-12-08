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

use std::fs as std_fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rand::{RngCore, thread_rng};
use rustc_data_structures::base_n::{BaseNString, CASE_INSENSITIVE, ToBaseN};
use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_data_structures::svh::Svh;
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_data_structures::{base_n, flock};
use rustc_errors::ErrorGuaranteed;
use rustc_fs_util::{LinkOrCopy, link_or_copy, try_canonicalize};
use rustc_middle::bug;
use rustc_session::config::CrateType;
use rustc_session::output::{collect_crate_types, find_crate_name};
use rustc_session::{Session, StableCrateId};
use tracing::debug;

use crate::errors;

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

/// Returns the path to a session's dependency graph.
pub(crate) fn dep_graph_path(sess: &Session) -> PathBuf {
    in_incr_comp_dir_sess(sess, DEP_GRAPH_FILENAME)
}

/// Returns the path to a session's staging dependency graph.
///
/// On the difference between dep-graph and staging dep-graph,
/// see `build_dep_graph`.
pub(crate) fn staging_dep_graph_path(sess: &Session) -> PathBuf {
    in_incr_comp_dir_sess(sess, STAGING_DEP_GRAPH_FILENAME)
}

pub(crate) fn work_products_path(sess: &Session) -> PathBuf {
    in_incr_comp_dir_sess(sess, WORK_PRODUCTS_FILENAME)
}

/// Returns the path to a session's query cache.
pub(crate) fn query_cache_path(sess: &Session) -> PathBuf {
    in_incr_comp_dir_sess(sess, QUERY_CACHE_FILENAME)
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
/// in the current session.
pub fn in_incr_comp_dir_sess(sess: &Session, file_name: &str) -> PathBuf {
    in_incr_comp_dir(&sess.incr_comp_session_dir(), file_name)
}

/// Returns the path for a given filename within the incremental compilation directory,
/// not necessarily from the current session.
///
/// To ensure the file is part of the current session, use [`in_incr_comp_dir_sess`].
pub fn in_incr_comp_dir(incr_comp_session_dir: &Path, file_name: &str) -> PathBuf {
    incr_comp_session_dir.join(file_name)
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
pub(crate) fn prepare_session_directory(sess: &Session) -> Result<(), ErrorGuaranteed> {
    if sess.opts.incremental.is_none() {
        return Ok(());
    }

    let _timer = sess.timer("incr_comp_prepare_session_directory");

    debug!("prepare_session_directory");

    // {incr-comp-dir}/{crate-name-and-disambiguator}
    let crate_dir = crate_path(sess);
    debug!("crate-dir: {}", crate_dir.display());
    create_dir(sess, &crate_dir, "crate")?;

    // Hack: canonicalize the path *after creating the directory*
    // because, on windows, long paths can cause problems;
    // canonicalization inserts this weird prefix that makes windows
    // tolerate long paths.
    let crate_dir = match try_canonicalize(&crate_dir) {
        Ok(v) => v,
        Err(err) => {
            return Err(sess.dcx().emit_err(errors::CanonicalizePath { path: crate_dir, err }));
        }
    };

    let mut source_directories_already_tried = FxHashSet::default();

    loop {
        // Generate a session directory of the form:
        //
        // {incr-comp-dir}/{crate-name-and-disambiguator}/s-{timestamp}-{random}-working
        let session_dir = generate_session_dir_path(&crate_dir);
        debug!("session-dir: {}", session_dir.display());

        // Lock the new session directory. If this fails, return an
        // error without retrying
        let (directory_lock, lock_file_path) = lock_directory(sess, &session_dir)?;

        // Now that we have the lock, we can actually create the session
        // directory
        create_dir(sess, &session_dir, "session")?;

        // Find a suitable source directory to copy from. Ignore those that we
        // have already tried before.
        let source_directory = find_source_directory(&crate_dir, &source_directories_already_tried);

        let Some(source_directory) = source_directory else {
            // There's nowhere to copy from, we're done
            debug!(
                "no source directory found. Continuing with empty session \
                    directory."
            );

            sess.init_incr_comp_session(session_dir, directory_lock);
            return Ok(());
        };

        debug!("attempting to copy data from source: {}", source_directory.display());

        // Try copying over all files from the source directory
        if let Ok(allows_links) = copy_files(sess, &session_dir, &source_directory) {
            debug!("successfully copied data from: {}", source_directory.display());

            if !allows_links {
                sess.dcx().emit_warn(errors::HardLinkFailed { path: &session_dir });
            }

            sess.init_incr_comp_session(session_dir, directory_lock);
            return Ok(());
        } else {
            debug!("copying failed - trying next directory");

            // Something went wrong while trying to copy/link files from the
            // source directory. Try again with a different one.
            source_directories_already_tried.insert(source_directory);

            // Try to remove the session directory we just allocated. We don't
            // know if there's any garbage in it from the failed copy action.
            if let Err(err) = safe_remove_dir_all(&session_dir) {
                sess.dcx().emit_warn(errors::DeletePartial { path: &session_dir, err });
            }

            delete_session_dir_lock_file(sess, &lock_file_path);
            drop(directory_lock);
        }
    }
}

/// This function finalizes and thus 'publishes' the session directory by
/// renaming it to `s-{timestamp}-{svh}` and releasing the file lock.
/// If there have been compilation errors, however, this function will just
/// delete the presumably invalid session directory.
pub fn finalize_session_directory(sess: &Session, svh: Option<Svh>) {
    if sess.opts.incremental.is_none() {
        return;
    }
    // The svh is always produced when incr. comp. is enabled.
    let svh = svh.unwrap();

    let _timer = sess.timer("incr_comp_finalize_session_directory");

    let incr_comp_session_dir: PathBuf = sess.incr_comp_session_dir().clone();

    if sess.dcx().has_errors_or_delayed_bugs().is_some() {
        // If there have been any errors during compilation, we don't want to
        // publish this session directory. Rather, we'll just delete it.

        debug!(
            "finalize_session_directory() - invalidating session directory: {}",
            incr_comp_session_dir.display()
        );

        if let Err(err) = safe_remove_dir_all(&*incr_comp_session_dir) {
            sess.dcx().emit_warn(errors::DeleteFull { path: &incr_comp_session_dir, err });
        }

        let lock_file_path = lock_file_path(&*incr_comp_session_dir);
        delete_session_dir_lock_file(sess, &lock_file_path);
        sess.mark_incr_comp_session_as_invalid();
    }

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

    match rename_path_with_retry(&*incr_comp_session_dir, &new_path, 3) {
        Ok(_) => {
            debug!("finalize_session_directory() - directory renamed successfully");

            // This unlocks the directory
            sess.finalize_incr_comp_session(new_path);
        }
        Err(e) => {
            // Warn about the error. However, no need to abort compilation now.
            sess.dcx().emit_warn(errors::Finalize { path: &incr_comp_session_dir, err: e });

            debug!("finalize_session_directory() - error, marking as invalid");
            // Drop the file lock, so we can garage collect
            sess.mark_incr_comp_session_as_invalid();
        }
    }

    let _ = garbage_collect_session_directories(sess);
}

pub(crate) fn delete_all_session_dir_contents(sess: &Session) -> io::Result<()> {
    let sess_dir_iterator = sess.incr_comp_session_dir().read_dir()?;
    for entry in sess_dir_iterator {
        let entry = entry?;
        safe_remove_file(&entry.path())?
    }
    Ok(())
}

fn copy_files(sess: &Session, target_dir: &Path, source_dir: &Path) -> Result<bool, ()> {
    // We acquire a shared lock on the lock file of the directory, so that
    // nobody deletes it out from under us while we are reading from it.
    let lock_file_path = lock_file_path(source_dir);

    // not exclusive
    let Ok(_lock) = flock::Lock::new(
        &lock_file_path,
        false, // don't wait,
        false, // don't create
        false,
    ) else {
        // Could not acquire the lock, don't try to copy from here
        return Err(());
    };

    let Ok(source_dir_iterator) = source_dir.read_dir() else {
        return Err(());
    };

    let mut files_linked = 0;
    let mut files_copied = 0;

    for entry in source_dir_iterator {
        match entry {
            Ok(entry) => {
                let file_name = entry.file_name();

                let target_file_path = target_dir.join(file_name);
                let source_path = entry.path();

                debug!("copying into session dir: {}", source_path.display());
                match link_or_copy(source_path, target_file_path) {
                    Ok(LinkOrCopy::Link) => files_linked += 1,
                    Ok(LinkOrCopy::Copy) => files_copied += 1,
                    Err(_) => return Err(()),
                }
            }
            Err(_) => return Err(()),
        }
    }

    if sess.opts.unstable_opts.incremental_info {
        eprintln!(
            "[incremental] session directory: \
                  {files_linked} files hard-linked"
        );
        eprintln!(
            "[incremental] session directory: \
                 {files_copied} files copied"
        );
    }

    Ok(files_linked > 0 || files_copied == 0)
}

/// Generates unique directory path of the form:
/// {crate_dir}/s-{timestamp}-{random-number}-working
fn generate_session_dir_path(crate_dir: &Path) -> PathBuf {
    let timestamp = timestamp_to_string(SystemTime::now());
    debug!("generate_session_dir_path: timestamp = {}", timestamp);
    let random_number = thread_rng().next_u32();
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

fn create_dir(sess: &Session, path: &Path, dir_tag: &str) -> Result<(), ErrorGuaranteed> {
    match std_fs::create_dir_all(path) {
        Ok(()) => {
            debug!("{} directory created successfully", dir_tag);
            Ok(())
        }
        Err(err) => Err(sess.dcx().emit_err(errors::CreateIncrCompDir { tag: dir_tag, path, err })),
    }
}

/// Allocate the lock-file and lock it.
fn lock_directory(
    sess: &Session,
    session_dir: &Path,
) -> Result<(flock::Lock, PathBuf), ErrorGuaranteed> {
    let lock_file_path = lock_file_path(session_dir);
    debug!("lock_directory() - lock_file: {}", lock_file_path.display());

    match flock::Lock::new(
        &lock_file_path,
        false, // don't wait
        true,  // create the lock file
        true,
    ) {
        // the lock should be exclusive
        Ok(lock) => Ok((lock, lock_file_path)),
        Err(lock_err) => {
            let is_unsupported_lock = flock::Lock::error_unsupported(&lock_err);
            Err(sess.dcx().emit_err(errors::CreateLock {
                lock_err,
                session_dir,
                is_unsupported_lock,
                is_cargo: rustc_session::utils::was_invoked_from_cargo(),
            }))
        }
    }
}

fn delete_session_dir_lock_file(sess: &Session, lock_file_path: &Path) {
    if let Err(err) = safe_remove_file(lock_file_path) {
        sess.dcx().emit_warn(errors::DeleteLock { path: lock_file_path, err });
    }
}

/// Finds the most recent published session directory that is not in the
/// ignore-list.
fn find_source_directory(
    crate_dir: &Path,
    source_directories_already_tried: &FxHashSet<PathBuf>,
) -> Option<PathBuf> {
    let iter = crate_dir
        .read_dir()
        .unwrap() // FIXME
        .filter_map(|e| e.ok().map(|e| e.path()));

    find_source_directory_in_iter(iter, source_directories_already_tried)
}

fn find_source_directory_in_iter<I>(
    iter: I,
    source_directories_already_tried: &FxHashSet<PathBuf>,
) -> Option<PathBuf>
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

        if source_directories_already_tried.contains(&session_dir)
            || !is_session_directory(&directory_name)
            || !is_finalized(&directory_name)
        {
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

fn crate_path(sess: &Session) -> PathBuf {
    let incr_dir = sess.opts.incremental.as_ref().unwrap().clone();

    let crate_name = find_crate_name(sess, &[]);
    let crate_types = collect_crate_types(sess, &[]);
    let stable_crate_id = StableCrateId::new(
        crate_name,
        crate_types.contains(&CrateType::Executable),
        sess.opts.cg.metadata.clone(),
        sess.cfg_version,
    );

    let crate_name =
        format!("{crate_name}-{}", stable_crate_id.as_u64().to_base_fixed_len(CASE_INSENSITIVE));
    incr_dir.join(crate_name)
}

fn is_old_enough_to_be_collected(timestamp: SystemTime) -> bool {
    timestamp < SystemTime::now() - Duration::from_secs(10)
}

/// Runs garbage collection for the current session.
pub(crate) fn garbage_collect_session_directories(sess: &Session) -> io::Result<()> {
    debug!("garbage_collect_session_directories() - begin");

    let session_directory = sess.incr_comp_session_dir();
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
            if let Err(err) = safe_remove_dir_all(&path) {
                sess.dcx().emit_warn(errors::InvalidGcFailed { path: &path, err });
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
                match flock::Lock::new(
                    &lock_file_path,
                    false, // don't wait
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
                        return Some((
                            (timestamp, crate_directory.join(directory_name)),
                            Some(lock),
                        ));
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
                match flock::Lock::new(
                    &lock_file_path,
                    false, // don't wait
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
    let deletion_candidates = deletion_candidates.into();

    // Delete all but the most recent of the candidates
    all_except_most_recent(deletion_candidates).into_items().all(|(path, lock)| {
        debug!("garbage_collect_session_directories() - deleting `{}`", path.display());

        if let Err(err) = safe_remove_dir_all(&path) {
            sess.dcx().emit_warn(errors::FinalizedGcFailed { path: &path, err });
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

    if let Err(err) = safe_remove_dir_all(path) {
        sess.dcx().emit_warn(errors::SessionGcFailed { path, err });
    } else {
        delete_session_dir_lock_file(sess, &lock_file_path(path));
    }
}

fn all_except_most_recent(
    deletion_candidates: UnordMap<(SystemTime, PathBuf), Option<flock::Lock>>,
) -> UnordMap<PathBuf, Option<flock::Lock>> {
    let most_recent = deletion_candidates.items().map(|(&(timestamp, _), _)| timestamp).max();

    if let Some(most_recent) = most_recent {
        deletion_candidates
            .into_items()
            .filter(|&((timestamp, _), _)| timestamp != most_recent)
            .map(|((_, path), lock)| (path, lock))
            .collect()
    } else {
        UnordMap::default()
    }
}

/// Since paths of artifacts within session directories can get quite long, we
/// need to support deleting files with very long paths. The regular
/// WinApi functions only support paths up to 260 characters, however. In order
/// to circumvent this limitation, we canonicalize the path of the directory
/// before passing it to std::fs::remove_dir_all(). This will convert the path
/// into the '\\?\' format, which supports much longer paths.
fn safe_remove_dir_all(p: &Path) -> io::Result<()> {
    let canonicalized = match try_canonicalize(p) {
        Ok(canonicalized) => canonicalized,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };

    std_fs::remove_dir_all(canonicalized)
}

fn safe_remove_file(p: &Path) -> io::Result<()> {
    let canonicalized = match try_canonicalize(p) {
        Ok(canonicalized) => canonicalized,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };

    match std_fs::remove_file(canonicalized) {
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
