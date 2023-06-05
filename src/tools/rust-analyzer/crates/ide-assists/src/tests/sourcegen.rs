//! Generates `assists.md` documentation.

use std::{fmt, fs, path::Path};

use test_utils::project_root;

#[test]
fn sourcegen_assists_docs() {
    let assists = Assist::collect();

    {
        // Generate doctests.

        let mut buf = "
use super::check_doc_test;
"
        .to_string();
        for assist in assists.iter() {
            for (idx, section) in assist.sections.iter().enumerate() {
                let test_id =
                    if idx == 0 { assist.id.clone() } else { format!("{}_{idx}", &assist.id) };
                let test = format!(
                    r######"
#[test]
fn doctest_{}() {{
    check_doc_test(
        "{}",
r#####"
{}"#####, r#####"
{}"#####)
}}
"######,
                    &test_id,
                    &assist.id,
                    reveal_hash_comments(&section.before),
                    reveal_hash_comments(&section.after)
                );

                buf.push_str(&test)
            }
        }
        let buf = sourcegen::add_preamble("sourcegen_assists_docs", sourcegen::reformat(buf));
        sourcegen::ensure_file_contents(
            &project_root().join("crates/ide-assists/src/tests/generated.rs"),
            &buf,
        );
    }

    {
        // Generate assists manual. Note that we do _not_ commit manual to the
        // git repo. Instead, `cargo xtask release` runs this test before making
        // a release.

        let contents = sourcegen::add_preamble(
            "sourcegen_assists_docs",
            assists.into_iter().map(|it| it.to_string()).collect::<Vec<_>>().join("\n\n"),
        );
        let dst = project_root().join("docs/user/generated_assists.adoc");
        fs::write(dst, contents).unwrap();
    }
}

#[derive(Debug)]
struct Section {
    doc: String,
    before: String,
    after: String,
}

#[derive(Debug)]
struct Assist {
    id: String,
    location: sourcegen::Location,
    sections: Vec<Section>,
}

impl Assist {
    fn collect() -> Vec<Assist> {
        let handlers_dir = project_root().join("crates/ide-assists/src/handlers");

        let mut res = Vec::new();
        for path in sourcegen::list_rust_files(&handlers_dir) {
            collect_file(&mut res, path.as_path());
        }
        res.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
        return res;

        fn collect_file(acc: &mut Vec<Assist>, path: &Path) {
            let text = fs::read_to_string(path).unwrap();
            let comment_blocks = sourcegen::CommentBlock::extract("Assist", &text);

            for block in comment_blocks {
                let id = block.id;
                assert!(
                    id.chars().all(|it| it.is_ascii_lowercase() || it == '_'),
                    "invalid assist id: {id:?}"
                );
                let mut lines = block.contents.iter().peekable();
                let location = sourcegen::Location { file: path.to_path_buf(), line: block.line };
                let mut assist = Assist { id, location, sections: Vec::new() };

                while lines.peek().is_some() {
                    let doc = take_until(lines.by_ref(), "```").trim().to_string();
                    assert!(
                        (doc.chars().next().unwrap().is_ascii_uppercase() && doc.ends_with('.'))
                            || assist.sections.len() > 0,
                        "\n\n{}: assist docs should be proper sentences, with capitalization and a full stop at the end.\n\n{}\n\n",
                        &assist.id,
                        doc,
                    );

                    let before = take_until(lines.by_ref(), "```");

                    assert_eq!(lines.next().unwrap().as_str(), "->");
                    assert_eq!(lines.next().unwrap().as_str(), "```");
                    let after = take_until(lines.by_ref(), "```");

                    assist.sections.push(Section { doc, before, after });
                }

                acc.push(assist)
            }
        }

        fn take_until<'a>(lines: impl Iterator<Item = &'a String>, marker: &str) -> String {
            let mut buf = Vec::new();
            for line in lines {
                if line == marker {
                    break;
                }
                buf.push(line.clone());
            }
            buf.join("\n")
        }
    }
}

impl fmt::Display for Assist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = writeln!(
            f,
            "[discrete]\n=== `{}`
**Source:** {}",
            self.id, self.location,
        );

        for section in &self.sections {
            let before = section.before.replace("$0", "┃"); // Unicode pseudo-graphics bar
            let after = section.after.replace("$0", "┃");
            let _ = writeln!(
                f,
                "
{}

.Before
```rust
{}```

.After
```rust
{}```",
                section.doc,
                hide_hash_comments(&before),
                hide_hash_comments(&after)
            );
        }

        Ok(())
    }
}

fn hide_hash_comments(text: &str) -> String {
    text.split('\n') // want final newline
        .filter(|&it| !(it.starts_with("# ") || it == "#"))
        .map(|it| format!("{it}\n"))
        .collect()
}

fn reveal_hash_comments(text: &str) -> String {
    text.split('\n') // want final newline
        .map(|it| {
            if let Some(stripped) = it.strip_prefix("# ") {
                stripped
            } else if it == "#" {
                ""
            } else {
                it
            }
        })
        .map(|it| format!("{it}\n"))
        .collect()
}
