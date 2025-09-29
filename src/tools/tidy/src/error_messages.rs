use std::collections::HashMap;
use std::path::Path;

use fluent_syntax::ast::{Entry, Expression, InlineExpression, PatternElement};
use fluent_syntax::parser;
use regex::Regex;

use crate::diagnostics::{CheckId, DiagCtx, RunningCheck};

pub fn check(root_path: &Path, diag_ctx: DiagCtx) {
    let path = &root_path.join("compiler");
    let mut check = diag_ctx.start_check(CheckId::new("error_messages").path(path));
    let regex = Regex::new(r#"(?:(?:#\[(diag|label|suggestion|note|help|help_once|multipart_suggestion)\()|(?:fluent(_generated)?::))\s*(?<id>[a-z_][a-zA-Z0-9_]+)\s*[\n,\)};]"#).unwrap();

    for dir in std::fs::read_dir(path).unwrap() {
        let Ok(dir) = dir else { continue };
        let dir = dir.path();
        let messages_file = dir.join("messages.ftl");

        if messages_file.is_file() {
            check_if_messages_are_used(&mut check, &dir.join("src"), &messages_file, &regex);
        }
    }
}

fn check_fluent_ast<'a>(ids: &mut HashMap<&'a str, bool>, elem: &PatternElement<&'a str>) {
    if let PatternElement::Placeable { expression } = elem {
        match expression {
            Expression::Inline(InlineExpression::MessageReference { id, .. }) => {
                *ids.entry(&id.name).or_default() = true;
            }
            Expression::Select { variants, .. } => {
                for variant in variants {
                    for elem in &variant.value.elements {
                        check_fluent_ast(ids, elem);
                    }
                }
            }
            _ => {}
        }
    }
}

fn check_if_messages_are_used(
    check: &mut RunningCheck,
    src_path: &Path,
    messages_file: &Path,
    regex: &Regex,
) {
    // First we retrieve all error messages ID.
    let content = std::fs::read_to_string(messages_file).expect("failed to read file");
    let resource =
        parser::parse(content.as_str()).expect("Errors encountered while parsing a resource.");

    let mut ids: HashMap<&str, bool> = HashMap::new();
    for entry in &resource.body {
        if let Entry::Message(msg) = entry {
            let id: &str = &msg.id.name;
            if !ids.contains_key(&id) {
                ids.insert(id, false);
            }
            if let Some(value) = &msg.value {
                for elem in &value.elements {
                    check_fluent_ast(&mut ids, elem);
                }
            }
            for attr in &msg.attributes {
                for elem in &attr.value.elements {
                    check_fluent_ast(&mut ids, elem);
                }
            }
        }
    }

    assert!(!ids.is_empty());

    let skip = |f: &Path, is_dir: bool| !is_dir && !f.extension().is_some_and(|ext| ext == "rs");
    crate::walk::walk(src_path, skip, &mut |_path: &_, content: &str| {
        for cap in regex.captures_iter(content) {
            let id = &cap["id"];
            if let Some(found) = ids.get_mut(id) {
                // Error message IDs can be used more than once.
                *found = true;
            }
        }
    });
    const TO_IGNORE: &[&str] = &[
        // FIXME: #114050
        "hir_typeck_option_result_asref",
    ];
    for (id, found) in ids {
        if !found && !TO_IGNORE.iter().any(|to_ignore| *to_ignore == id) {
            check.error(format!("unused message ID `{id}` from `{}`", messages_file.display()));
        }
    }
}
