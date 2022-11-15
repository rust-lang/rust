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
    Root { childs: Vec<Node<L>> },
    Directory { name: PathBuf, childs: Vec<Node<L>>, license: Option<L> },
    File { name: PathBuf, license: L },
    FileGroup { names: Vec<PathBuf>, license: L },
    Empty,
}

impl Node<LicenseId> {
    pub(crate) fn simplify(&mut self) {
        self.merge_directories();
        self.collapse_in_licensed_directories();
        self.merge_directory_licenses();
        self.merge_file_groups();
        self.remove_empty();
    }

    /// Initially, trees are built by the build() function with each file practically having a
    /// separate directory tree, like so:
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
            Node::Root { childs } | Node::Directory { childs, license: None, .. } => {
                let mut directories = BTreeMap::new();
                let mut files = Vec::new();

                for child in childs.drain(..) {
                    match child {
                        Node::Directory { name, mut childs, license: None } => {
                            directories.entry(name).or_insert_with(Vec::new).append(&mut childs);
                        }
                        file @ Node::File { .. } => {
                            files.push(file);
                        }
                        Node::Empty => {}
                        Node::Root { .. } => {
                            panic!("can't have a root inside another element");
                        }
                        Node::FileGroup { .. } => {
                            panic!("FileGroup should not be present at this stage");
                        }
                        Node::Directory { license: Some(_), .. } => {
                            panic!("license should not be set at this stage");
                        }
                    }
                }

                childs.extend(directories.into_iter().map(|(name, childs)| Node::Directory {
                    name,
                    childs,
                    license: None,
                }));
                childs.append(&mut files);

                for child in &mut *childs {
                    child.merge_directories();
                }
            }
            Node::Empty => {}
            Node::File { .. } => {}
            Node::FileGroup { .. } => {
                panic!("FileGroup should not be present at this stage");
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
            Node::Directory { childs, license, .. } => {
                for child in &mut *childs {
                    child.collapse_in_licensed_directories();
                }

                let mut licenses_count = BTreeMap::new();
                for child in &*childs {
                    let Some(license) = child.license() else { continue };
                    *licenses_count.entry(license).or_insert(0) += 1;
                }

                let most_popular_license = licenses_count
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(license, _)| license);

                if let Some(most_popular_license) = most_popular_license {
                    childs.retain(|child| child.license() != Some(most_popular_license));
                    *license = Some(most_popular_license);
                }
            }
            Node::Root { childs } => {
                for child in &mut *childs {
                    child.collapse_in_licensed_directories();
                }
            }
            Node::File { .. } => {}
            Node::FileGroup { .. } => {}
            Node::Empty => {}
        }
    }

    /// Reduce the depth of the tree by merging subdirectories with the same license as their
    /// parent directory into their parent, and adjusting the paths of the childs accordingly.
    fn merge_directory_licenses(&mut self) {
        match self {
            Node::Root { childs } => {
                for child in &mut *childs {
                    child.merge_directory_licenses();
                }
            }
            Node::Directory { childs, license, .. } => {
                let mut to_add = Vec::new();
                for child in &mut *childs {
                    child.merge_directory_licenses();

                    let Node::Directory {
                        name: child_name,
                        childs: child_childs,
                        license: child_license,
                    } = child else { continue };

                    if child_license != license {
                        continue;
                    }
                    for mut child_child in child_childs.drain(..) {
                        match &mut child_child {
                            Node::Root { .. } => {
                                panic!("can't have a root inside another element");
                            }
                            Node::FileGroup { .. } => {
                                panic!("FileGroup should not be present at this stage");
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
                childs.append(&mut to_add);
            }
            Node::Empty => {}
            Node::File { .. } => {}
            Node::FileGroup { .. } => {}
        }
    }

    /// This pass groups multiple files in a directory with the same license into a single
    /// "FileGroup", so that the license of all those files can be reported as a group.
    ///
    /// Crucially this pass runs after collapse_in_licensed_directories, so the most common license
    /// will already be marked as the directory's license and won't be turned into a group.
    fn merge_file_groups(&mut self) {
        match self {
            Node::Root { childs } | Node::Directory { childs, .. } => {
                let mut grouped = BTreeMap::new();

                for child in &mut *childs {
                    child.merge_file_groups();
                    if let Node::File { name, license } = child {
                        grouped.entry(*license).or_insert_with(Vec::new).push(name.clone());
                        *child = Node::Empty;
                    }
                }

                for (license, mut names) in grouped.into_iter() {
                    if names.len() == 1 {
                        childs.push(Node::File { license, name: names.pop().unwrap() });
                    } else {
                        childs.push(Node::FileGroup { license, names });
                    }
                }
            }
            Node::File { .. } => {}
            Node::FileGroup { .. } => panic!("FileGroup should not be present at this stage"),
            Node::Empty => {}
        }
    }

    /// Some nodes were replaced with Node::Empty to mark them for deletion. As the last step, make
    /// sure to remove them from the tree.
    fn remove_empty(&mut self) {
        match self {
            Node::Root { childs } | Node::Directory { childs, .. } => {
                for child in &mut *childs {
                    child.remove_empty();
                }
                childs.retain(|child| !matches!(child, Node::Empty));
            }
            Node::FileGroup { .. } => {}
            Node::File { .. } => {}
            Node::Empty => {}
        }
    }

    fn license(&self) -> Option<LicenseId> {
        match self {
            Node::Directory { childs, license: Some(license), .. } if childs.is_empty() => {
                Some(*license)
            }
            Node::File { license, .. } => Some(*license),
            _ => None,
        }
    }
}

pub(crate) fn build(mut input: Vec<(PathBuf, LicenseId)>) -> Node<LicenseId> {
    let mut childs = Vec::new();

    // Ensure reproducibility of all future steps.
    input.sort();

    for (path, license) in input {
        let mut node = Node::File { name: path.file_name().unwrap().into(), license };
        for component in path.parent().unwrap_or_else(|| Path::new(".")).components().rev() {
            node = Node::Directory {
                name: component.as_os_str().into(),
                childs: vec![node],
                license: None,
            };
        }

        childs.push(node);
    }

    Node::Root { childs }
}

pub(crate) fn strip_interning(
    node: Node<LicenseId>,
    interner: &LicensesInterner,
) -> Node<&License> {
    match node {
        Node::Root { childs } => Node::Root {
            childs: childs.into_iter().map(|child| strip_interning(child, interner)).collect(),
        },
        Node::Directory { name, childs, license } => Node::Directory {
            childs: childs.into_iter().map(|child| strip_interning(child, interner)).collect(),
            license: license.map(|license| interner.resolve(license)),
            name,
        },
        Node::File { name, license } => Node::File { name, license: interner.resolve(license) },
        Node::FileGroup { names, license } => {
            Node::FileGroup { names, license: interner.resolve(license) }
        }
        Node::Empty => Node::Empty,
    }
}
