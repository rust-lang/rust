//! Path resolution engine for the VFS.
//!
//! Implements iterative resolution of absolute paths with support for:
//! - Multi-component paths: `/a/b/c`
//! - Current-directory component (`.`): ignored
//! - Parent-directory component (`..`): pops the last resolved component
//! - Mount-point crossings: delegated to the global mount table
//! - Symbolic link following: up to [`MAX_SYMLINK_DEPTH`] levels
//!
//! # Design
//! Rather than building a full in-kernel `dentry` cache, this engine
//! normalises the path string first and then passes the cleaned path to
//! [`crate::vfs::mount::lookup`].  This is intentionally simple for ACT III:
//! a richer cache can be layered in later.
//!
//! The public entry points are:
//! - [`resolve`] — resolve with symlink following (for `open`, `stat`, etc.)
//! - [`resolve_no_follow`] — resolve the path without following the final symlink
//!   (for `readlink`, `lstat`-style operations)

use abi::errors::{Errno, SysResult};
use alloc::string::String;

/// Maximum number of components allowed in a path before returning `ENAMETOOLONG`.
const MAX_COMPONENTS: usize = 64;

/// Maximum number of symlink expansions before returning `ELOOP`.
const MAX_SYMLINK_DEPTH: usize = 40;

/// Resolve an absolute path to a VFS node, following symlinks.
///
/// Normalises `path` (collapsing `.` and `..` components) and delegates
/// to the mount table for the final lookup.  Symlinks encountered during
/// path traversal are expanded iteratively up to [`MAX_SYMLINK_DEPTH`] times.
///
/// # Errors
/// - `EINVAL`  — `path` is not absolute (does not start with `/`).
/// - `ENAMETOOLONG` — too many path components.
/// - `ENOENT`  — the path does not resolve to any mounted node.
/// - `ELOOP`   — too many levels of symbolic links.
pub fn resolve(path: &str) -> SysResult<alloc::sync::Arc<dyn crate::vfs::VfsNode>> {
    resolve_at(path, 0)
}

/// Resolve an absolute path **without** following the final component if it is
/// a symlink.  Symlinks in intermediate path components are still followed.
///
/// Used by `readlink` and `lstat`-style callers that want to inspect the
/// symlink itself rather than its target.
pub fn resolve_no_follow(path: &str) -> SysResult<alloc::sync::Arc<dyn crate::vfs::VfsNode>> {
    let normalised = normalise(path)?;
    // Walk all but the last component with symlink following, then do a plain
    // lookup for the last component.
    let components: alloc::vec::Vec<&str> = normalised[1..]
        .split('/')
        .filter(|c| !c.is_empty())
        .collect();

    if components.is_empty() {
        // Path is "/".
        return crate::vfs::mount::lookup("/");
    }

    // Resolve all intermediate components (with symlink following).
    if components.len() > 1 {
        let parent: String = {
            let mut s = String::from("/");
            for (i, c) in components[..components.len() - 1].iter().enumerate() {
                if i > 0 {
                    s.push('/');
                }
                s.push_str(c);
            }
            s
        };
        // Ensure intermediate directories exist and are reachable (follows symlinks in parent).
        let _ = resolve_at(&parent, 0)?;
    }

    // Look up the final component without following it.
    crate::vfs::mount::lookup(&normalised)
}

/// Internal helper that resolves `path` starting at a given symlink-follow depth.
fn resolve_at(path: &str, depth: usize) -> SysResult<alloc::sync::Arc<dyn crate::vfs::VfsNode>> {
    if depth > MAX_SYMLINK_DEPTH {
        return Err(Errno::ELOOP);
    }

    let normalised = normalise(path)?;

    // Walk path component by component so we can follow symlinks at each step.
    let components: alloc::vec::Vec<&str> = normalised[1..]
        .split('/')
        .filter(|c| !c.is_empty())
        .collect();

    if components.is_empty() {
        // Root directory — never a symlink.
        return crate::vfs::mount::lookup("/");
    }

    let mut current_path = String::with_capacity(normalised.len());
    for (i, component) in components.iter().enumerate() {
        current_path.push('/');
        current_path.push_str(component);

        let node = crate::vfs::mount::lookup(&current_path)?;

        // Check if this node is a symlink.
        match node.readlink() {
            Ok(target) => {
                // Compute the path after following this symlink:
                // remaining components after the current one.
                let remaining: String = components[i + 1..].join("/");

                let new_base = if target.starts_with('/') {
                    target
                } else {
                    // Relative symlink: resolve relative to the parent directory.
                    let parent = match current_path.rfind('/') {
                        Some(0) => String::from("/"),
                        Some(idx) => String::from(&current_path[..idx]),
                        None => String::from("/"),
                    };
                    if parent == "/" {
                        alloc::format!("/{}", target)
                    } else {
                        alloc::format!("{}/{}", parent, target)
                    }
                };

                let new_path = if remaining.is_empty() {
                    new_base
                } else {
                    alloc::format!("{}/{}", new_base, remaining)
                };

                // Recursively resolve with incremented depth.
                return resolve_at(&new_path, depth + 1);
            }
            Err(_) => {
                // Not a symlink.  If this is an intermediate component we must
                // verify it is a directory before continuing.
                if i < components.len() - 1 {
                    let stat = node.stat()?;
                    if !stat.is_dir() {
                        return Err(Errno::ENOTDIR);
                    }
                }
            }
        }
    }

    // All components have been walked; return the final node.
    crate::vfs::mount::lookup(&current_path)
}

/// Normalise an absolute path, resolving `.` and `..` components.
///
/// Returns the canonical absolute path string.
///
/// # Examples
/// - `/a/./b` → `/a/b`
/// - `/a/b/../c` → `/a/c`
/// - `/a/b/../../c` → `/c`
/// - `/../..` → `/` (cannot go above root)
pub fn normalise(path: &str) -> SysResult<String> {
    if path.is_empty() || !path.starts_with('/') {
        return Err(Errno::EINVAL);
    }

    let mut components: alloc::vec::Vec<&str> = alloc::vec::Vec::new();

    for component in path.split('/') {
        match component {
            "" | "." => {
                // Skip empty segments (consecutive slashes) and current-dir.
            }
            ".." => {
                // Go up — ignore if already at root.
                components.pop();
            }
            name => {
                if components.len() >= MAX_COMPONENTS {
                    return Err(Errno::ENAMETOOLONG);
                }
                components.push(name);
            }
        }
    }

    if components.is_empty() {
        return Ok(String::from("/"));
    }

    // Exact capacity: one '/' per component plus each component's length.
    let capacity: usize = components.iter().map(|c| c.len() + 1).sum();
    let mut result = String::with_capacity(capacity);
    for c in &components {
        result.push('/');
        result.push_str(c);
    }
    Ok(result)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_path() {
        assert_eq!(normalise("/a/b/c").unwrap(), "/a/b/c");
    }

    #[test]
    fn test_dot_components_removed() {
        assert_eq!(normalise("/a/./b").unwrap(), "/a/b");
        assert_eq!(normalise("/./a").unwrap(), "/a");
    }

    #[test]
    fn test_dot_dot_goes_up() {
        assert_eq!(normalise("/a/b/../c").unwrap(), "/a/c");
        assert_eq!(normalise("/a/b/../../c").unwrap(), "/c");
    }

    #[test]
    fn test_dot_dot_at_root_stays() {
        assert_eq!(normalise("/../..").unwrap(), "/");
        assert_eq!(normalise("/..").unwrap(), "/");
    }

    #[test]
    fn test_root_path() {
        assert_eq!(normalise("/").unwrap(), "/");
    }

    #[test]
    fn test_trailing_slash() {
        assert_eq!(normalise("/a/b/").unwrap(), "/a/b");
    }

    #[test]
    fn test_double_slash() {
        assert_eq!(normalise("//a//b").unwrap(), "/a/b");
    }

    #[test]
    fn test_relative_path_returns_einval() {
        assert!(matches!(normalise("relative/path"), Err(Errno::EINVAL)));
        assert!(matches!(normalise(""), Err(Errno::EINVAL)));
    }

    #[test]
    fn test_complex_normalisation() {
        assert_eq!(normalise("/a/b/c/../../d").unwrap(), "/a/d");
        assert_eq!(normalise("/a/./b/./c").unwrap(), "/a/b/c");
    }

    // ── Tests for realpath requirements ──────────────────────────────────────

    /// The canonical form of an already-absolute path must be stable (idempotent).
    #[test]
    fn test_realpath_idempotent() {
        let once = normalise("/usr/local/bin").unwrap();
        let twice = normalise(&once).unwrap();
        assert_eq!(once, twice);
    }

    /// Deeply nested `../` chains must collapse correctly.
    #[test]
    fn test_realpath_deep_dotdot_chain() {
        assert_eq!(normalise("/a/b/c/d/../../../../e").unwrap(), "/e");
        assert_eq!(normalise("/a/b/c/../../../d/../e").unwrap(), "/e");
    }

    /// Mixed `.` and `..` with redundant separators must yield a clean path.
    #[test]
    fn test_realpath_mixed_dot_components() {
        assert_eq!(normalise("/a/./b//../c").unwrap(), "/a/c");
    }

    /// Output must not contain trailing slashes (except for root).
    #[test]
    fn test_realpath_no_trailing_slash() {
        let result = normalise("/a/b/c/").unwrap();
        assert!(!result.ends_with('/') || result == "/");
    }
}
