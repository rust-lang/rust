//! ncurses-compatible database discovery.
//!
//! Does not support hashed database, only filesystem!

use std::env;
use std::fs;
use std::path::PathBuf;

#[cfg(test)]
mod tests;

/// Return path to database entry for `term`
#[allow(deprecated)]
pub(crate) fn get_dbpath_for_term(term: &str) -> Option<PathBuf> {
    let mut dirs_to_search = Vec::new();
    let first_char = term.chars().next()?;

    // Find search directory
    if let Some(dir) = env::var_os("TERMINFO") {
        dirs_to_search.push(PathBuf::from(dir));
    }

    if let Ok(dirs) = env::var("TERMINFO_DIRS") {
        for i in dirs.split(':') {
            if i == "" {
                dirs_to_search.push(PathBuf::from("/usr/share/terminfo"));
            } else {
                dirs_to_search.push(PathBuf::from(i));
            }
        }
    } else {
        // Found nothing in TERMINFO_DIRS, use the default paths:
        // According to  /etc/terminfo/README, after looking at
        // ~/.terminfo, ncurses will search /etc/terminfo, then
        // /lib/terminfo, and eventually /usr/share/terminfo.
        // On Haiku the database can be found at /boot/system/data/terminfo
        if let Some(mut homedir) = env::home_dir() {
            homedir.push(".terminfo");
            dirs_to_search.push(homedir)
        }

        dirs_to_search.push(PathBuf::from("/etc/terminfo"));
        dirs_to_search.push(PathBuf::from("/lib/terminfo"));
        dirs_to_search.push(PathBuf::from("/usr/share/terminfo"));
        dirs_to_search.push(PathBuf::from("/boot/system/data/terminfo"));
    }

    // Look for the terminal in all of the search directories
    for mut p in dirs_to_search {
        if fs::metadata(&p).is_ok() {
            p.push(&first_char.to_string());
            p.push(&term);
            if fs::metadata(&p).is_ok() {
                return Some(p);
            }
            p.pop();
            p.pop();

            // on some installations the dir is named after the hex of the char
            // (e.g., macOS)
            p.push(&format!("{:x}", first_char as usize));
            p.push(term);
            if fs::metadata(&p).is_ok() {
                return Some(p);
            }
        }
    }
    None
}
