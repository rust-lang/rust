//! Generates `assists.md` documentation.

use std::{fmt, fs, path::PathBuf};

use crate::{
    codegen::{self, extract_comment_blocks_with_empty_lines, Location, Mode, PREAMBLE},
    project_root, rust_files, Result,
};

pub fn generate_feature_docs(mode: Mode) -> Result<()> {
    let features = Feature::collect()?;
    let contents = features.into_iter().map(|it| it.to_string()).collect::<Vec<_>>().join("\n\n");
    let contents = format!("//{}\n{}\n", PREAMBLE, contents.trim());
    let dst = project_root().join("docs/user/generated_features.adoc");
    codegen::update(&dst, &contents, mode)?;
    Ok(())
}

#[derive(Debug)]
struct Feature {
    id: String,
    location: Location,
    doc: String,
}

impl Feature {
    fn collect() -> Result<Vec<Feature>> {
        let mut res = Vec::new();
        for path in rust_files(&project_root()) {
            collect_file(&mut res, path)?;
        }
        res.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
        return Ok(res);

        fn collect_file(acc: &mut Vec<Feature>, path: PathBuf) -> Result<()> {
            let text = fs::read_to_string(&path)?;
            let comment_blocks = extract_comment_blocks_with_empty_lines("Feature", &text);

            for block in comment_blocks {
                let id = block.id;
                assert!(is_valid_feature_name(&id), "invalid feature name: {:?}", id);
                let doc = block.contents.join("\n");
                let location = Location::new(path.clone(), block.line);
                acc.push(Feature { id, location, doc })
            }

            Ok(())
        }
    }
}

fn is_valid_feature_name(feature: &str) -> bool {
    'word: for word in feature.split_whitespace() {
        for &short in ["to", "and"].iter() {
            if word == short {
                continue 'word;
            }
        }
        for &short in ["To", "And"].iter() {
            if word == short {
                return false;
            }
        }
        if !word.starts_with(char::is_uppercase) {
            return false;
        }
    }
    true
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== {}\n**Source:** {}\n{}", self.id, self.location, self.doc)
    }
}
