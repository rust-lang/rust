//! Generates `assists.md` documentation.

use std::{fmt, fs, path::Path};

use crate::{
    codegen::{self, extract_comment_blocks_with_empty_lines, Location, Mode},
    project_root, rust_files, Result,
};

pub fn generate_assists_tests(mode: Mode) -> Result<()> {
    let assists = Assist::collect()?;
    generate_tests(&assists, mode)
}

pub fn generate_assists_docs(mode: Mode) -> Result<()> {
    let assists = Assist::collect()?;
    let contents = assists.into_iter().map(|it| it.to_string()).collect::<Vec<_>>().join("\n\n");
    let contents = contents.trim().to_string() + "\n";
    let dst = project_root().join("docs/user/generated_assists.adoc");
    codegen::update(&dst, &contents, mode)
}

#[derive(Debug)]
struct Assist {
    id: String,
    location: Location,
    doc: String,
    before: String,
    after: String,
}

impl Assist {
    fn collect() -> Result<Vec<Assist>> {
        let mut res = Vec::new();
        for path in rust_files(&project_root().join(codegen::ASSISTS_DIR)) {
            collect_file(&mut res, path.as_path())?;
        }
        res.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
        return Ok(res);

        fn collect_file(acc: &mut Vec<Assist>, path: &Path) -> Result<()> {
            let text = fs::read_to_string(path)?;
            let comment_blocks = extract_comment_blocks_with_empty_lines("Assist", &text);

            for block in comment_blocks {
                // FIXME: doesn't support blank lines yet, need to tweak
                // `extract_comment_blocks` for that.
                let id = block.id;
                assert!(
                    id.chars().all(|it| it.is_ascii_lowercase() || it == '_'),
                    "invalid assist id: {:?}",
                    id
                );
                let mut lines = block.contents.iter();

                let doc = take_until(lines.by_ref(), "```").trim().to_string();
                assert!(
                    doc.chars().next().unwrap().is_ascii_uppercase() && doc.ends_with('.'),
                    "\n\n{}: assist docs should be proper sentences, with capitalization and a full stop at the end.\n\n{}\n\n",
                    id, doc,
                );

                let before = take_until(lines.by_ref(), "```");

                assert_eq!(lines.next().unwrap().as_str(), "->");
                assert_eq!(lines.next().unwrap().as_str(), "```");
                let after = take_until(lines.by_ref(), "```");
                let location = Location::new(path.to_path_buf(), block.line);
                acc.push(Assist { id, location, doc, before, after })
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
            Ok(())
        }
    }
}

impl fmt::Display for Assist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let before = self.before.replace("<|>", "┃"); // Unicode pseudo-graphics bar
        let after = self.after.replace("<|>", "┃");
        writeln!(
            f,
            "[discrete]\n=== `{}`
**Source:** {}

{}

.Before
```rust
{}```

.After
```rust
{}```",
            self.id,
            self.location,
            self.doc,
            hide_hash_comments(&before),
            hide_hash_comments(&after)
        )
    }
}

fn generate_tests(assists: &[Assist], mode: Mode) -> Result<()> {
    let mut buf = String::from("use super::check_doc_test;\n");

    for assist in assists.iter() {
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
            assist.id,
            assist.id,
            reveal_hash_comments(&assist.before),
            reveal_hash_comments(&assist.after)
        );

        buf.push_str(&test)
    }
    let buf = crate::reformat(buf)?;
    codegen::update(&project_root().join(codegen::ASSISTS_TESTS), &buf, mode)
}

fn hide_hash_comments(text: &str) -> String {
    text.split('\n') // want final newline
        .filter(|&it| !(it.starts_with("# ") || it == "#"))
        .map(|it| format!("{}\n", it))
        .collect()
}

fn reveal_hash_comments(text: &str) -> String {
    text.split('\n') // want final newline
        .map(|it| {
            if it.starts_with("# ") {
                &it[2..]
            } else if it == "#" {
                ""
            } else {
                it
            }
        })
        .map(|it| format!("{}\n", it))
        .collect()
}
