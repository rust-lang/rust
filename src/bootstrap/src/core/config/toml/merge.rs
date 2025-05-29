//! Provides traits and implementations for merging configuration structures.
use std::collections::HashSet;
use std::path::PathBuf;

use build_helper::exit;

use super::ReplaceOpt;

pub(crate) trait Merge {
    fn merge(
        &mut self,
        parent_config_path: Option<PathBuf>,
        included_extensions: &mut HashSet<PathBuf>,
        other: Self,
        replace: ReplaceOpt,
    );
}

impl<T> Merge for Option<T> {
    fn merge(
        &mut self,
        _parent_config_path: Option<PathBuf>,
        _included_extensions: &mut HashSet<PathBuf>,
        other: Self,
        replace: ReplaceOpt,
    ) {
        match replace {
            ReplaceOpt::IgnoreDuplicate => {
                if self.is_none() {
                    *self = other;
                }
            }
            ReplaceOpt::Override => {
                if other.is_some() {
                    *self = other;
                }
            }
            ReplaceOpt::ErrorOnDuplicate => {
                if other.is_some() {
                    if self.is_some() {
                        if cfg!(test) {
                            panic!("overriding existing option")
                        } else {
                            eprintln!("overriding existing option");
                            exit!(2);
                        }
                    } else {
                        *self = other;
                    }
                }
            }
        }
    }
}
