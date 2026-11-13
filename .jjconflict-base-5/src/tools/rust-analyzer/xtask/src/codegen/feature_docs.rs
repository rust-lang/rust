//! Generates `features_generated.md` documentation.

use std::{fmt, fs, io, path::PathBuf};

use crate::{
    codegen::{CommentBlock, Location, add_preamble},
    project_root,
    util::list_rust_files,
};

pub(crate) fn generate(check: bool) {
    let features = Feature::collect().unwrap();
    // Do not generate docs when run with `--check`
    if check {
        return;
    }
    let contents = features.into_iter().map(|it| it.to_string()).collect::<Vec<_>>().join("\n\n");
    let contents = add_preamble(crate::flags::CodegenType::FeatureDocs, contents);
    let dst = project_root().join("docs/book/src/features_generated.md");
    fs::write(dst, contents).unwrap();
}

#[derive(Debug)]
struct Feature {
    id: String,
    location: Location,
    doc: String,
}

impl Feature {
    fn collect() -> io::Result<Vec<Feature>> {
        let crates_dir = project_root().join("crates");

        let mut res = Vec::new();
        for path in list_rust_files(&crates_dir) {
            collect_file(&mut res, path)?;
        }
        res.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
        return Ok(res);

        fn collect_file(acc: &mut Vec<Feature>, path: PathBuf) -> io::Result<()> {
            let text = std::fs::read_to_string(&path)?;
            let comment_blocks = CommentBlock::extract("Feature", &text);

            for block in comment_blocks {
                let id = block.id;
                if let Err(msg) = is_valid_feature_name(&id) {
                    panic!("invalid feature name: {id:?}:\n  {msg}")
                }
                let doc = block.contents.join("\n");
                let location = Location { file: path.clone(), line: block.line };
                acc.push(Feature { id, location, doc })
            }

            Ok(())
        }
    }
}

fn is_valid_feature_name(feature: &str) -> Result<(), String> {
    'word: for word in feature.split_whitespace() {
        for short in ["to", "and"] {
            if word == short {
                continue 'word;
            }
        }
        for short in ["To", "And"] {
            if word == short {
                return Err(format!("Don't capitalize {word:?}"));
            }
        }
        if !word.starts_with(char::is_uppercase) {
            return Err(format!("Capitalize {word:?}"));
        }
    }
    Ok(())
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "### {}\n**Source:** {}\n{}", self.id, self.location, self.doc)
    }
}
