use ignore::{self, gitignore};

use crate::config::{FileName, IgnoreList};

pub(crate) struct IgnorePathSet {
    ignore_set: gitignore::Gitignore,
}

impl IgnorePathSet {
    pub(crate) fn from_ignore_list(ignore_list: &IgnoreList) -> Result<Self, ignore::Error> {
        let mut ignore_builder = gitignore::GitignoreBuilder::new(ignore_list.rustfmt_toml_path());

        for ignore_path in ignore_list {
            ignore_builder.add_line(None, ignore_path.to_str().unwrap())?;
        }

        Ok(IgnorePathSet {
            ignore_set: ignore_builder.build()?,
        })
    }

    pub(crate) fn is_match(&self, file_name: &FileName) -> bool {
        match file_name {
            FileName::Stdin => false,
            FileName::Real(p) => self
                .ignore_set
                .matched_path_or_any_parents(p, false)
                .is_ignore(),
        }
    }
}

#[cfg(test)]
mod test {
    use rustfmt_config_proc_macro::nightly_only_test;

    #[nightly_only_test]
    #[test]
    fn test_ignore_path_set() {
        use std::path::{Path, PathBuf};

        use crate::config::{Config, FileName};
        use crate::ignore_path::IgnorePathSet;
        let config =
            Config::from_toml(r#"ignore = ["foo.rs", "bar_dir/*"]"#, Path::new("")).unwrap();
        let ignore_path_set = IgnorePathSet::from_ignore_list(&config.ignore()).unwrap();

        assert!(ignore_path_set.is_match(&FileName::Real(PathBuf::from("src/foo.rs"))));
        assert!(ignore_path_set.is_match(&FileName::Real(PathBuf::from("bar_dir/baz.rs"))));
        assert!(!ignore_path_set.is_match(&FileName::Real(PathBuf::from("src/bar.rs"))));
    }
}
