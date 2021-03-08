//! Generates `assists.md` documentation.

use std::{fmt, path::PathBuf};

use xshell::write_file;

use crate::{
    codegen::{extract_comment_blocks_with_empty_lines, Location, PREAMBLE},
    project_root, rust_files, Result,
};

pub(crate) fn generate_feature_docs() -> Result<()> {
    let features = Feature::collect()?;
    let contents = features.into_iter().map(|it| it.to_string()).collect::<Vec<_>>().join("\n\n");
    let contents = format!("//{}\n{}\n", PREAMBLE, contents.trim());
    let dst = project_root().join("docs/user/generated_features.adoc");
    write_file(&dst, &contents)?;
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
        for path in rust_files() {
            collect_file(&mut res, path)?;
        }
        res.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
        return Ok(res);

        fn collect_file(acc: &mut Vec<Feature>, path: PathBuf) -> Result<()> {
            let text = xshell::read_file(&path)?;
            let comment_blocks = extract_comment_blocks_with_empty_lines("Feature", &text);

            for block in comment_blocks {
                let id = block.id;
                if let Err(msg) = is_valid_feature_name(&id) {
                    panic!("invalid feature name: {:?}:\n  {}", id, msg)
                }
                let doc = block.contents.join("\n");
                let location = Location::new(path.clone(), block.line);
                acc.push(Feature { id, location, doc })
            }

            Ok(())
        }
    }
}

fn is_valid_feature_name(feature: &str) -> Result<(), String> {
    'word: for word in feature.split_whitespace() {
        for &short in ["to", "and"].iter() {
            if word == short {
                continue 'word;
            }
        }
        for &short in ["To", "And"].iter() {
            if word == short {
                return Err(format!("Don't capitalize {:?}", word));
            }
        }
        if !word.starts_with(char::is_uppercase) {
            return Err(format!("Capitalize {:?}", word));
        }
    }
    Ok(())
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== {}\n**Source:** {}\n{}", self.id, self.location, self.doc)
    }
}
