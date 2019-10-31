use std::{fs, path::Path};

use crate::{
    codegen::{self, extract_comment_blocks_with_empty_lines, Mode},
    project_root, Result,
};

pub fn generate_assists_docs(mode: Mode) -> Result<()> {
    let assists = collect_assists()?;
    generate_tests(&assists, mode)?;
    generate_docs(&assists, mode)?;
    Ok(())
}

#[derive(Debug)]
struct Assist {
    id: String,
    doc: String,
    before: String,
    after: String,
}

fn collect_assists() -> Result<Vec<Assist>> {
    let mut res = Vec::new();
    for entry in fs::read_dir(project_root().join(codegen::ASSISTS_DIR))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            collect_file(&mut res, path.as_path())?;
        }
    }
    res.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
    return Ok(res);

    fn collect_file(acc: &mut Vec<Assist>, path: &Path) -> Result<()> {
        let text = fs::read_to_string(path)?;
        let comment_blocks = extract_comment_blocks_with_empty_lines(&text);

        for block in comment_blocks {
            // FIXME: doesn't support blank lines yet, need to tweak
            // `extract_comment_blocks` for that.
            let mut lines = block.iter();
            let first_line = lines.next().unwrap();
            if !first_line.starts_with("Assist: ") {
                continue;
            }
            let id = first_line["Assist: ".len()..].to_string();
            assert!(
                id.chars().all(|it| it.is_ascii_lowercase() || it == '_'),
                "invalid assist id: {:?}",
                id
            );

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
            acc.push(Assist { id, doc, before, after })
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

fn generate_tests(assists: &[Assist], mode: Mode) -> Result<()> {
    let mut buf = String::from("use super::check;\n");

    for assist in assists.iter() {
        let test = format!(
            r######"
#[test]
fn doctest_{}() {{
    check(
        "{}",
r#####"
{}
"#####, r#####"
{}
"#####)
}}
"######,
            assist.id, assist.id, assist.before, assist.after
        );

        buf.push_str(&test)
    }
    let buf = codegen::reformat(buf)?;
    codegen::update(&project_root().join(codegen::ASSISTS_TESTS), &buf, mode)
}

fn generate_docs(assists: &[Assist], mode: Mode) -> Result<()> {
    let mut buf = String::from(
        "# Assists\n\nCursor position or selection is signified by `┃` character.\n\n",
    );

    for assist in assists {
        let before = assist.before.replace("<|>", "┃"); // Unicode pseudo-graphics bar
        let after = assist.after.replace("<|>", "┃");
        let docs = format!(
            "
## `{}`

{}

```rust
// BEFORE
{}

// AFTER
{}
```
",
            assist.id, assist.doc, before, after
        );
        buf.push_str(&docs);
    }

    codegen::update(&project_root().join(codegen::ASSISTS_DOCS), &buf, mode)
}
