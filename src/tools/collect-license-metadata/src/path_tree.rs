//! Tools like REUSE output per-file licensing information, but we need to condense it in the
//! minimum amount of data that still represents the same licensing metadata. This module is
//! responsible for that, by turning the list of paths into a tree and executing simplification
//! passes over the tree to remove redundant information.

use crate::licenses::{License, LicenseId, LicensesInterner};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(serde::Serialize)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub(crate) enum Node<L> {
    Root { children: Vec<Node<L>> },
    Directory { name: PathBuf, children: Vec<Node<L>>, license: Option<L> },
    File { name: PathBuf, license: L },
    Group { files: Vec<PathBuf>, directories: Vec<PathBuf>, license: L },
    Empty,
}

impl Node<LicenseId> {
    pub(crate) fn simplify(&mut self) {
        self.merge_directories();
        self.collapse_in_licensed_directories();
        self.merge_directory_licenses();
        self.merge_groups();
        self.remove_empty();
    }

    /// Initially, the build() function constructs a list of separate paths from the file
    /// system root down to each file, like so:
    ///
    /// ```text
    ///         ┌─► ./ ──► compiler/ ──► rustc/ ──► src/ ──► main.rs
    ///         │
    /// <root> ─┼─► ./ ──► compiler/ ──► rustc/ ──► Cargo.toml
    ///         │
    ///         └─► ./ ──► library/ ───► std/ ──► Cargo.toml
    /// ```
    ///
    /// This pass is responsible for turning that into a proper directory tree:
    ///
    /// ```text
    ///                 ┌─► compiler/ ──► rustc/ ──┬─► src/ ──► main.rs
    ///                 │                          │
    /// <root> ──► ./ ──┤                          └─► Cargo.toml
    ///                 │
    ///                 └─► library/ ───► std/ ──► Cargo.toml
    /// ```
    fn merge_directories(&mut self) {
        match self {
            Node::Root { children } | Node::Directory { children, license: None, .. } => {
                let mut directories = BTreeMap::new();
                let mut files = Vec::new();

                for child in children.drain(..) {
                    match child {
                        Node::Directory { name, mut children, license: None } => {
                            directories.entry(name).or_insert_with(Vec::new).append(&mut children);
                        }
                        file @ Node::File { .. } => {
                            files.push(file);
                        }
                        Node::Empty => {}
                        Node::Root { .. } => {
                            panic!("can't have a root inside another element");
                        }
                        Node::Group { .. } => {
                            panic!("Group should not be present at this stage");
                        }
                        Node::Directory { license: Some(_), .. } => {
                            panic!("license should not be set at this stage");
                        }
                    }
                }

                children.extend(directories.into_iter().map(|(name, children)| Node::Directory {
                    name,
                    children,
                    license: None,
                }));
                children.append(&mut files);

                for child in &mut *children {
                    child.merge_directories();
                }
            }
            Node::Empty => {}
            Node::File { .. } => {}
            Node::Group { .. } => {
                panic!("Group should not be present at this stage");
            }
            Node::Directory { license: Some(_), .. } => {
                panic!("license should not be set at this stage");
            }
        }
    }

    /// In our codebase, most files in a directory have the same license as the other files in that
    /// same directory, so it's redundant to store licensing metadata for all the files. Instead,
    /// we can add a license for a whole directory, and only record the exceptions to a directory
    /// licensing metadata.
    ///
    /// We cannot instead record only the difference to Rust's standard licensing, as the majority
    /// of the files in our repository are *not* licensed under Rust's standard licensing due to
    /// our inclusion of LLVM.
    fn collapse_in_licensed_directories(&mut self) {
        match self {
            Node::Directory { children, license, .. } => {
                for child in &mut *children {
                    child.collapse_in_licensed_directories();
                }

                let mut licenses_count = BTreeMap::new();
                for child in &*children {
                    let Some(license) = child.license() else { continue };
                    *licenses_count.entry(license).or_insert(0) += 1;
                }

                let most_popular_license = licenses_count
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(license, _)| license);

                if let Some(most_popular_license) = most_popular_license {
                    children.retain(|child| child.license() != Some(most_popular_license));
                    *license = Some(most_popular_license);
                }
            }
            Node::Root { children } => {
                for child in &mut *children {
                    child.collapse_in_licensed_directories();
                }
            }
            Node::File { .. } => {}
            Node::Group { .. } => panic!("group should not be present at this stage"),
            Node::Empty => {}
        }
    }

    /// Reduce the depth of the tree by merging subdirectories with the same license as their
    /// parent directory into their parent, and adjusting the paths of the children accordingly.
    fn merge_directory_licenses(&mut self) {
        match self {
            Node::Root { children } => {
                for child in &mut *children {
                    child.merge_directory_licenses();
                }
            }
            Node::Directory { children, license, .. } => {
                let mut to_add = Vec::new();
                for child in &mut *children {
                    child.merge_directory_licenses();

                    let Node::Directory {
                        name: child_name,
                        children: child_children,
                        license: child_license,
                    } = child
                    else {
                        continue;
                    };

                    if child_license != license {
                        continue;
                    }
                    for mut child_child in child_children.drain(..) {
                        match &mut child_child {
                            Node::Root { .. } => {
                                panic!("can't have a root inside another element");
                            }
                            Node::Group { .. } => {
                                panic!("Group should not be present at this stage");
                            }
                            Node::Directory { name: child_child_name, .. } => {
                                *child_child_name = child_name.join(&child_child_name);
                            }
                            Node::File { name: child_child_name, .. } => {
                                *child_child_name = child_name.join(&child_child_name);
                            }
                            Node::Empty => {}
                        }
                        to_add.push(child_child);
                    }

                    *child = Node::Empty;
                }
                children.append(&mut to_add);
            }
            Node::Empty => {}
            Node::File { .. } => {}
            Node::Group { .. } => panic!("Group should not be present at this stage"),
        }
    }

    /// This pass groups multiple files in a directory with the same license into a single
    /// "Group", so that the license of all those files can be reported as a group.
    ///
    /// This also merges directories *without exceptions*.
    ///
    /// Crucially this pass runs after collapse_in_licensed_directories, so the most common license
    /// will already be marked as the directory's license and won't be turned into a group.
    fn merge_groups(&mut self) {
        #[derive(Default)]
        struct Grouped {
            files: Vec<PathBuf>,
            directories: Vec<PathBuf>,
        }
        match self {
            Node::Root { children } | Node::Directory { children, .. } => {
                let mut grouped: BTreeMap<LicenseId, Grouped> = BTreeMap::new();

                for child in &mut *children {
                    child.merge_groups();
                    match child {
                        Node::Directory { name, children, license: Some(license) } => {
                            if children.is_empty() {
                                grouped
                                    .entry(*license)
                                    .or_insert_with(Grouped::default)
                                    .directories
                                    .push(name.clone());
                                *child = Node::Empty;
                            }
                        }
                        Node::File { name, license } => {
                            grouped
                                .entry(*license)
                                .or_insert_with(Grouped::default)
                                .files
                                .push(name.clone());
                            *child = Node::Empty;
                        }
                        _ => {}
                    }
                }

                for (license, mut grouped) in grouped.into_iter() {
                    if grouped.files.len() + grouped.directories.len() <= 1 {
                        if let Some(name) = grouped.files.pop() {
                            children.push(Node::File { license, name });
                        } else if let Some(name) = grouped.directories.pop() {
                            children.push(Node::Directory {
                                name,
                                children: Vec::new(),
                                license: Some(license),
                            });
                        }
                    } else {
                        children.push(Node::Group {
                            license,
                            files: grouped.files,
                            directories: grouped.directories,
                        });
                    }
                }
            }
            Node::File { .. } => {}
            Node::Group { .. } => panic!("FileGroup should not be present at this stage"),
            Node::Empty => {}
        }
    }

    /// Some nodes were replaced with Node::Empty to mark them for deletion. As the last step, make
    /// sure to remove them from the tree.
    fn remove_empty(&mut self) {
        match self {
            Node::Root { children } | Node::Directory { children, .. } => {
                for child in &mut *children {
                    child.remove_empty();
                }
                children.retain(|child| !matches!(child, Node::Empty));
            }
            Node::Group { .. } => {}
            Node::File { .. } => {}
            Node::Empty => {}
        }
    }

    fn license(&self) -> Option<LicenseId> {
        match self {
            Node::Directory { children, license: Some(license), .. } if children.is_empty() => {
                Some(*license)
            }
            Node::File { license, .. } => Some(*license),
            _ => None,
        }
    }
}

pub(crate) fn build(mut input: Vec<(PathBuf, LicenseId)>) -> Node<LicenseId> {
    let mut children = Vec::new();

    // Ensure reproducibility of all future steps.
    input.sort();

    for (path, license) in input {
        let mut node = Node::File { name: path.file_name().unwrap().into(), license };
        for component in path.parent().unwrap_or_else(|| Path::new(".")).components().rev() {
            node = Node::Directory {
                name: component.as_os_str().into(),
                children: vec![node],
                license: None,
            };
        }

        children.push(node);
    }

    Node::Root { children }
}

/// Convert a `Node<LicenseId>` into a `Node<&License>`, expanding all interned license IDs with a
/// reference to the actual license metadata.
pub(crate) fn expand_interned_licenses(
    node: Node<LicenseId>,
    interner: &LicensesInterner,
) -> Node<&License> {
    match node {
        Node::Root { children } => Node::Root {
            children: children
                .into_iter()
                .map(|child| expand_interned_licenses(child, interner))
                .collect(),
        },
        Node::Directory { name, children, license } => Node::Directory {
            children: children
                .into_iter()
                .map(|child| expand_interned_licenses(child, interner))
                .collect(),
            license: license.map(|license| interner.resolve(license)),
            name,
        },
        Node::File { name, license } => Node::File { name, license: interner.resolve(license) },
        Node::Group { files, directories, license } => {
            Node::Group { files, directories, license: interner.resolve(license) }
        }
        Node::Empty => Node::Empty,
    }
}
