//! `ra_vfs_glob` crate implements exclusion rules for vfs.
//!
//! By default, we include only `.rs` files, and skip some know offenders like
//! `/target` or `/node_modules` altogether.
//!
//! It's also possible to add custom exclusion globs.

use globset::{GlobSet, GlobSetBuilder};
use ra_vfs::{Filter, RelativePath};

pub use globset::{Glob, GlobBuilder};

const ALWAYS_IGNORED: &[&str] = &["target/**", "**/node_modules/**", "**/.git/**"];
const IGNORED_FOR_NON_MEMBERS: &[&str] = &["examples/**", "tests/**", "benches/**"];

pub struct RustPackageFilterBuilder {
    is_member: bool,
    exclude: GlobSetBuilder,
}

impl Default for RustPackageFilterBuilder {
    fn default() -> RustPackageFilterBuilder {
        RustPackageFilterBuilder { is_member: false, exclude: GlobSetBuilder::new() }
    }
}

impl RustPackageFilterBuilder {
    pub fn set_member(mut self, is_member: bool) -> RustPackageFilterBuilder {
        self.is_member = is_member;
        self
    }
    pub fn exclude(mut self, glob: Glob) -> RustPackageFilterBuilder {
        self.exclude.add(glob);
        self
    }
    pub fn into_vfs_filter(self) -> Box<dyn Filter> {
        let RustPackageFilterBuilder { is_member, mut exclude } = self;
        for &glob in ALWAYS_IGNORED {
            exclude.add(Glob::new(glob).unwrap());
        }
        if !is_member {
            for &glob in IGNORED_FOR_NON_MEMBERS {
                exclude.add(Glob::new(glob).unwrap());
            }
        }
        Box::new(RustPackageFilter { exclude: exclude.build().unwrap() })
    }
}

struct RustPackageFilter {
    exclude: GlobSet,
}

impl Filter for RustPackageFilter {
    fn include_dir(&self, dir_path: &RelativePath) -> bool {
        !self.exclude.is_match(dir_path.as_str())
    }

    fn include_file(&self, file_path: &RelativePath) -> bool {
        file_path.extension() == Some("rs")
    }
}

#[test]
fn test_globs() {
    let filter = RustPackageFilterBuilder::default().set_member(true).into_vfs_filter();

    assert!(filter.include_dir(RelativePath::new("src/tests")));
    assert!(filter.include_dir(RelativePath::new("src/target")));
    assert!(filter.include_dir(RelativePath::new("tests")));
    assert!(filter.include_dir(RelativePath::new("benches")));

    assert!(!filter.include_dir(RelativePath::new("target")));
    assert!(!filter.include_dir(RelativePath::new("src/foo/.git")));
    assert!(!filter.include_dir(RelativePath::new("foo/node_modules")));

    let filter = RustPackageFilterBuilder::default().set_member(false).into_vfs_filter();

    assert!(filter.include_dir(RelativePath::new("src/tests")));
    assert!(filter.include_dir(RelativePath::new("src/target")));

    assert!(!filter.include_dir(RelativePath::new("target")));
    assert!(!filter.include_dir(RelativePath::new("src/foo/.git")));
    assert!(!filter.include_dir(RelativePath::new("foo/node_modules")));
    assert!(!filter.include_dir(RelativePath::new("tests")));
    assert!(!filter.include_dir(RelativePath::new("benches")));

    let filter = RustPackageFilterBuilder::default()
        .set_member(true)
        .exclude(Glob::new("src/llvm-project/**").unwrap())
        .into_vfs_filter();

    assert!(!filter.include_dir(RelativePath::new("src/llvm-project/clang")));
}
