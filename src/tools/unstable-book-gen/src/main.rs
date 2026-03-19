//! Auto-generate stub docs for the unstable book

use std::collections::BTreeSet;
use std::env;
use std::fs::{self, write};
use std::path::Path;

use tidy::diagnostics::RunningCheck;
use tidy::features::{
    Feature, Features, Status, collect_env_vars, collect_lang_features, collect_lib_features,
};
use tidy::t;
use tidy::unstable_book::{
    COMPILER_FLAGS_DIR, ENV_VARS_DIR, LANG_FEATURES_DIR, LIB_FEATURES_DIR, PATH_STR,
    collect_unstable_book_section_file_names, collect_unstable_feature_names,
};

fn generate_stub_issue(path: &Path, name: &str, issue: u32, description: &str) {
    let content = format!(
        include_str!("stub-issue.md"),
        name = name,
        issue = issue,
        description = description
    );
    t!(write(path, content), path);
}

fn generate_stub_no_issue(path: &Path, name: &str, description: &str) {
    let content = format!(include_str!("stub-no-issue.md"), name = name, description = description);
    t!(write(path, content), path);
}

fn generate_stub_env_var(path: &Path, name: &str) {
    let content = format!(include_str!("stub-env-var.md"), name = name);
    t!(write(path, content), path);
}

fn set_to_summary_str(set: &BTreeSet<String>, dir: &str) -> String {
    set.iter()
        .map(|ref n| format!("    - [{}]({}/{}.md)", n.replace('-', "_"), dir, n))
        .fold("".to_owned(), |s, a| s + &a + "\n")
}

fn generate_summary(path: &Path, lang_features: &Features, lib_features: &Features) {
    let compiler_flags = collect_unstable_book_section_file_names(&path.join("src/compiler-flags"));
    let compiler_env_vars =
        collect_unstable_book_section_file_names(&path.join("src/compiler-environment-variables"));

    let compiler_flags_str = set_to_summary_str(&compiler_flags, "compiler-flags");
    let compiler_env_vars_str =
        set_to_summary_str(&compiler_env_vars, "compiler-environment-variables");

    let unstable_lang_features = collect_unstable_feature_names(&lang_features);
    let unstable_lib_features = collect_unstable_feature_names(&lib_features);

    let lang_features_str = set_to_summary_str(&unstable_lang_features, "language-features");
    let lib_features_str = set_to_summary_str(&unstable_lib_features, "library-features");

    let summary_path = path.join("src/SUMMARY.md");
    let content = format!(
        include_str!("SUMMARY.md"),
        compiler_env_vars = compiler_env_vars_str,
        compiler_flags = compiler_flags_str,
        language_features = lang_features_str,
        library_features = lib_features_str
    );
    t!(write(&summary_path, content), summary_path);
}

fn generate_feature_files(src: &Path, out: &Path, features: &Features) {
    let unstable_features = collect_unstable_feature_names(features);
    let unstable_section_file_names = collect_unstable_book_section_file_names(src);
    t!(fs::create_dir_all(&out));
    for feature_name in &unstable_features - &unstable_section_file_names {
        let feature_name_underscore = feature_name.replace('-', "_");
        let file_name = format!("{feature_name}.md");
        let out_file_path = out.join(&file_name);
        let feature = &features[&feature_name_underscore];
        let description = feature.description.as_deref().unwrap_or_default();

        if let Some(issue) = feature.tracking_issue {
            generate_stub_issue(
                &out_file_path,
                &feature_name_underscore,
                issue.get(),
                &description,
            );
        } else {
            generate_stub_no_issue(&out_file_path, &feature_name_underscore, &description);
        }
    }
}

fn generate_env_files(src: &Path, out: &Path, env_vars: &BTreeSet<String>) {
    let env_var_file_names = collect_unstable_book_section_file_names(src);
    t!(fs::create_dir_all(&out));
    for env_var in env_vars - &env_var_file_names {
        let file_name = format!("{env_var}.md");
        let out_file_path = out.join(&file_name);
        generate_stub_env_var(&out_file_path, &env_var);
    }
}

fn copy_recursive(from: &Path, to: &Path) {
    for entry in t!(fs::read_dir(from)) {
        let e = t!(entry);
        let t = t!(e.metadata());
        let dest = &to.join(e.file_name());
        if t.is_file() {
            t!(fs::copy(&e.path(), dest));
        } else if t.is_dir() {
            t!(fs::create_dir_all(dest));
            copy_recursive(&e.path(), dest);
        }
    }
}

fn collect_compiler_flags(compiler_path: &Path) -> Features {
    let options_path = compiler_path.join("rustc_session/src/options.rs");
    let options_rs = t!(fs::read_to_string(&options_path), options_path);
    parse_compiler_flags(&options_rs, &options_path)
}

const DESCRIPTION_FIELD: usize = 3;
const REQUIRED_FIELDS: usize = 4;
const OPTIONAL_FIELDS: usize = 5;

struct SourceBlock<'a> {
    content: &'a str,
    offset: usize,
}

struct ParsedOptionEntry<'a> {
    name: &'a str,
    name_start: usize,
    description: Option<String>,
    next_idx: usize,
}

fn parse_compiler_flags(options_rs: &str, options_path: &Path) -> Features {
    let options_block = find_options_block(options_rs, "UnstableOptions");
    let alphabetical_section = find_tidy_alphabetical_section(options_block);
    let section_line_offset = line_number(options_rs, alphabetical_section.offset) - 1;

    let mut features = Features::new();
    let mut idx = 0;

    while idx < alphabetical_section.content.len() {
        skip_ws_comments_and_attrs(alphabetical_section.content, &mut idx);
        if idx >= alphabetical_section.content.len() {
            break;
        }

        let entry = parse_one_entry(alphabetical_section.content, idx);
        idx = entry.next_idx;

        if entry.name == "help" {
            continue;
        }

        features.insert(
            entry.name.to_owned(),
            Feature {
                level: Status::Unstable,
                since: None,
                has_gate_test: false,
                tracking_issue: None,
                file: options_path.to_path_buf(),
                line: section_line_offset
                    + line_number(alphabetical_section.content, entry.name_start),
                description: entry.description,
            },
        );
    }

    features
}

fn parse_one_entry(source: &str, start_idx: usize) -> ParsedOptionEntry<'_> {
    let name_start = start_idx;
    let name_end =
        parse_ident_end(source, name_start).expect("expected an option name in UnstableOptions");
    let name = &source[name_start..name_end];
    let mut idx = name_end;

    skip_ws_comments(source, &mut idx);
    expect_byte(source, idx, b':', &format!("expected `:` after option name `{name}`"));
    idx += 1;

    idx =
        find_char_outside_nested(source, idx, b'=').expect("expected `=` in UnstableOptions entry");
    idx += 1;

    skip_ws_comments(source, &mut idx);
    expect_byte(source, idx, b'(', &format!("expected tuple payload for option `{name}`"));

    let tuple_start = idx;
    let tuple_end = find_matching_delimiter(source, tuple_start, b'(', b')')
        .expect("UnstableOptions tuple should be balanced");
    let next_idx = skip_past_entry_delimiter(source, tuple_end + 1, name);

    let description = if name == "help" {
        None
    } else {
        let fields = split_top_level_fields(&source[tuple_start + 1..tuple_end]);
        validate_option_fields(&fields, name);
        // The `options!` macro layout is `(init, parse, [dep_tracking...], desc, ...)`.
        Some(parse_string_literal(
            fields.get(DESCRIPTION_FIELD).expect("option description should be present"),
        ))
    };

    ParsedOptionEntry { name, name_start, description, next_idx }
}

fn find_options_block<'a>(source: &'a str, struct_name: &str) -> SourceBlock<'a> {
    let mut search_from = 0;

    while let Some(relative_start) = source[search_from..].find("options!") {
        let macro_start = search_from + relative_start;
        let open_brace = source[macro_start..]
            .find('{')
            .map(|relative| macro_start + relative)
            .expect("options! invocation should contain `{`");
        let close_brace = find_matching_delimiter(source, open_brace, b'{', b'}')
            .expect("options! invocation should have a matching `}`");
        let block = &source[open_brace + 1..close_brace];

        if block.trim_start().starts_with(struct_name) {
            return SourceBlock { content: block, offset: open_brace + 1 };
        }

        search_from = close_brace + 1;
    }

    panic!("could not find `{struct_name}` options! block");
}

fn find_tidy_alphabetical_section(block: SourceBlock<'_>) -> SourceBlock<'_> {
    let start_marker = "// tidy-alphabetical-start";
    let end_marker = "// tidy-alphabetical-end";

    let section_start = block
        .content
        .find(start_marker)
        .map(|start| start + start_marker.len())
        .expect("options! block should contain `// tidy-alphabetical-start`");
    let section_end = block.content[section_start..]
        .find(end_marker)
        .map(|end| section_start + end)
        .expect("options! block should contain `// tidy-alphabetical-end`");

    SourceBlock {
        content: &block.content[section_start..section_end],
        offset: block.offset + section_start,
    }
}

fn line_number(source: &str, offset: usize) -> usize {
    source[..offset].bytes().filter(|&byte| byte == b'\n').count() + 1
}

fn expect_byte(source: &str, idx: usize, expected: u8, context: &str) {
    assert_eq!(source.as_bytes().get(idx).copied(), Some(expected), "{context}");
}

fn skip_ws_comments_and_attrs(source: &str, idx: &mut usize) {
    loop {
        skip_ws_comments(source, idx);

        if source[*idx..].starts_with("#[") {
            let attr_start = *idx + 1;
            let attr_end = find_matching_delimiter(source, attr_start, b'[', b']')
                .expect("attribute should have matching `]`");
            *idx = attr_end + 1;
            continue;
        }

        break;
    }
}

fn skip_ws_comments(source: &str, idx: &mut usize) {
    loop {
        while let Some(byte) = source.as_bytes().get(*idx) {
            if byte.is_ascii_whitespace() {
                *idx += 1;
            } else {
                break;
            }
        }

        if source[*idx..].starts_with("//") {
            *idx = source[*idx..].find('\n').map_or(source.len(), |end| *idx + end + 1);
            continue;
        }

        if source[*idx..].starts_with("/*") {
            *idx = skip_block_comment(source, *idx);
            continue;
        }

        break;
    }
}

fn skip_block_comment(source: &str, mut idx: usize) -> usize {
    let mut depth = 1;
    idx += 2;

    while idx < source.len() {
        match source.as_bytes().get(idx..idx + 2) {
            Some(b"/*") => {
                depth += 1;
                idx += 2;
            }
            Some(b"*/") => {
                depth -= 1;
                idx += 2;
                if depth == 0 {
                    return idx;
                }
            }
            _ => idx += 1,
        }
    }

    panic!("unterminated block comment");
}

fn parse_ident_end(source: &str, start: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let first = *bytes.get(start)?;
    if !(first == b'_' || first.is_ascii_alphabetic()) {
        return None;
    }

    let mut idx = start + 1;
    while let Some(byte) = bytes.get(idx) {
        if *byte == b'_' || byte.is_ascii_alphanumeric() {
            idx += 1;
        } else {
            break;
        }
    }

    Some(idx)
}

fn find_char_outside_nested(source: &str, start: usize, needle: u8) -> Option<usize> {
    let mut idx = start;
    let mut paren_depth = 0;
    let mut bracket_depth = 0;
    let mut brace_depth = 0;

    while idx < source.len() {
        match source.as_bytes()[idx] {
            b'/' if source[idx..].starts_with("//") => {
                idx = source[idx..].find('\n').map_or(source.len(), |end| idx + end + 1);
            }
            b'/' if source[idx..].starts_with("/*") => idx = skip_block_comment(source, idx),
            b'"' => idx = skip_string_literal(source, idx),
            b'(' => {
                paren_depth += 1;
                idx += 1;
            }
            b')' => {
                paren_depth -= 1;
                idx += 1;
            }
            b'[' => {
                bracket_depth += 1;
                idx += 1;
            }
            b']' => {
                bracket_depth -= 1;
                idx += 1;
            }
            b'{' => {
                brace_depth += 1;
                idx += 1;
            }
            b'}' => {
                brace_depth -= 1;
                idx += 1;
            }
            byte if byte == needle
                && paren_depth == 0
                && bracket_depth == 0
                && brace_depth == 0 =>
            {
                return Some(idx);
            }
            _ => idx += 1,
        }
    }

    None
}

fn find_matching_delimiter(source: &str, start: usize, open: u8, close: u8) -> Option<usize> {
    let mut idx = start;
    let mut depth = 0;

    while idx < source.len() {
        match source.as_bytes()[idx] {
            b'/' if source[idx..].starts_with("//") => {
                idx = source[idx..].find('\n').map_or(source.len(), |end| idx + end + 1);
            }
            b'/' if source[idx..].starts_with("/*") => idx = skip_block_comment(source, idx),
            b'"' => idx = skip_string_literal(source, idx),
            byte if byte == open => {
                depth += 1;
                idx += 1;
            }
            byte if byte == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(idx);
                }
                idx += 1;
            }
            _ => idx += 1,
        }
    }

    None
}

fn split_top_level_fields(source: &str) -> Vec<&str> {
    let mut fields = Vec::new();
    let mut field_start = 0;
    let mut idx = 0;
    let mut paren_depth = 0;
    let mut bracket_depth = 0;
    let mut brace_depth = 0;

    while idx < source.len() {
        match source.as_bytes()[idx] {
            b'/' if source[idx..].starts_with("//") => {
                idx = source[idx..].find('\n').map_or(source.len(), |end| idx + end + 1);
            }
            b'/' if source[idx..].starts_with("/*") => idx = skip_block_comment(source, idx),
            b'"' => idx = skip_string_literal(source, idx),
            b'(' => {
                paren_depth += 1;
                idx += 1;
            }
            b')' => {
                paren_depth -= 1;
                idx += 1;
            }
            b'[' => {
                bracket_depth += 1;
                idx += 1;
            }
            b']' => {
                bracket_depth -= 1;
                idx += 1;
            }
            b'{' => {
                brace_depth += 1;
                idx += 1;
            }
            b'}' => {
                brace_depth -= 1;
                idx += 1;
            }
            b',' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                fields.push(source[field_start..idx].trim());
                idx += 1;
                field_start = idx;
            }
            _ => idx += 1,
        }
    }

    fields.push(source[field_start..].trim());
    fields
}

fn validate_option_fields(fields: &[&str], name: &str) {
    assert!(
        matches!(fields.len(), REQUIRED_FIELDS | OPTIONAL_FIELDS),
        "unexpected field count for option `{name}`: expected 4 or 5 fields, found {}",
        fields.len()
    );
    assert!(
        fields[2].starts_with('[') && fields[2].ends_with(']'),
        "expected dep-tracking field in option `{name}`, found `{}`",
        fields[2]
    );
    assert!(
        looks_like_string_literal(fields[DESCRIPTION_FIELD]),
        "expected description string literal in option `{name}`, found `{}`",
        fields[DESCRIPTION_FIELD]
    );

    if let Some(extra_field) = fields.get(REQUIRED_FIELDS) {
        assert!(
            extra_field.trim_start().starts_with("is_deprecated_and_do_nothing:"),
            "unexpected trailing field in option `{name}`: `{extra_field}`",
        );
    }
}

fn looks_like_string_literal(field: &str) -> bool {
    let field = field.trim();
    (field.starts_with('"') && field.ends_with('"')) || parse_raw_string_literal(field).is_some()
}

fn skip_past_entry_delimiter(source: &str, start: usize, name: &str) -> usize {
    let mut idx = start;
    skip_ws_comments(source, &mut idx);

    match source.as_bytes().get(idx).copied() {
        Some(b',') => idx + 1,
        None => idx,
        Some(byte) => {
            panic!("expected `,` after option entry `{name}`, found {:?}", char::from(byte))
        }
    }
}

fn skip_string_literal(source: &str, mut idx: usize) -> usize {
    idx += 1;

    while idx < source.len() {
        match source.as_bytes()[idx] {
            b'\\' => {
                idx += 1;
                if idx < source.len() {
                    idx += 1;
                }
            }
            b'"' => return idx + 1,
            _ => idx += 1,
        }
    }

    panic!("unterminated string literal");
}

fn parse_string_literal(literal: &str) -> String {
    let literal = literal.trim();

    if let Some(raw_literal) = parse_raw_string_literal(literal) {
        return raw_literal;
    }

    let inner = literal
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .expect("expected a string literal");
    let mut output = String::new();
    let mut chars = inner.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }

        let escaped = chars.next().expect("unterminated string escape");
        match escaped {
            '\n' => while chars.next_if(|ch| ch.is_whitespace()).is_some() {},
            '\r' => {
                let _ = chars.next_if_eq(&'\n');
                while chars.next_if(|ch| ch.is_whitespace()).is_some() {}
            }
            '"' => output.push('"'),
            '\'' => output.push('\''),
            '\\' => output.push('\\'),
            'n' => output.push('\n'),
            'r' => output.push('\r'),
            't' => output.push('\t'),
            '0' => output.push('\0'),
            'x' => {
                let hi = chars.next().expect("missing first hex digit in escape");
                let lo = chars.next().expect("missing second hex digit in escape");
                let byte = u8::from_str_radix(&format!("{hi}{lo}"), 16)
                    .expect("invalid hex escape in string literal");
                output.push(char::from(byte));
            }
            'u' => {
                assert_eq!(chars.next(), Some('{'), "expected `{{` after `\\u`");
                let mut digits = String::new();
                for ch in chars.by_ref() {
                    if ch == '}' {
                        break;
                    }
                    digits.push(ch);
                }
                let scalar =
                    u32::from_str_radix(&digits, 16).expect("invalid unicode escape in string");
                output.push(char::from_u32(scalar).expect("unicode escape should be valid"));
            }
            _ => panic!("unsupported escape in string literal"),
        }
    }

    output
}

fn parse_raw_string_literal(literal: &str) -> Option<String> {
    let rest = literal.strip_prefix('r')?;
    let hashes = rest.bytes().take_while(|&byte| byte == b'#').count();
    let quote_idx = 1 + hashes;

    if literal.as_bytes().get(quote_idx) != Some(&b'"') {
        return None;
    }

    let suffix = format!("\"{}", "#".repeat(hashes));
    let content = literal[quote_idx + 1..]
        .strip_suffix(&suffix)
        .expect("raw string literal should have a matching terminator");

    Some(content.to_owned())
}

fn main() {
    let library_path_str = env::args_os().nth(1).expect("library/ path required");
    let compiler_path_str = env::args_os().nth(2).expect("compiler/ path required");
    let src_path_str = env::args_os().nth(3).expect("src/ path required");
    let dest_path_str = env::args_os().nth(4).expect("destination path required");
    let library_path = Path::new(&library_path_str);
    let compiler_path = Path::new(&compiler_path_str);
    let src_path = Path::new(&src_path_str);
    let dest_path = Path::new(&dest_path_str);

    let lang_features = collect_lang_features(compiler_path, &mut RunningCheck::new_noop());
    let lib_features = collect_lib_features(library_path)
        .into_iter()
        .filter(|&(ref name, _)| !lang_features.contains_key(name))
        .collect();
    let env_vars = collect_env_vars(compiler_path);
    let compiler_flags = collect_compiler_flags(compiler_path);

    let doc_src_path = src_path.join(PATH_STR);

    t!(fs::create_dir_all(&dest_path));

    generate_feature_files(
        &doc_src_path.join(LANG_FEATURES_DIR),
        &dest_path.join(LANG_FEATURES_DIR),
        &lang_features,
    );
    generate_feature_files(
        &doc_src_path.join(LIB_FEATURES_DIR),
        &dest_path.join(LIB_FEATURES_DIR),
        &lib_features,
    );
    generate_feature_files(
        &doc_src_path.join(COMPILER_FLAGS_DIR),
        &dest_path.join(COMPILER_FLAGS_DIR),
        &compiler_flags,
    );
    generate_env_files(&doc_src_path.join(ENV_VARS_DIR), &dest_path.join(ENV_VARS_DIR), &env_vars);

    copy_recursive(&doc_src_path, &dest_path);

    generate_summary(&dest_path, &lang_features, &lib_features);
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{parse_compiler_flags, parse_one_entry, skip_ws_comments_and_attrs};

    #[test]
    fn parses_unstable_options_entries() {
        let options_rs = r#"
options! {
    UnstableOptions, UnstableOptionsTargetModifiers, Z_OPTIONS, dbopts, "Z", "unstable",

    // tidy-alphabetical-start
    #[rustc_lint_opt_deny_field_access("test attr")]
    allow_features: Option<Vec<String>> = (None, parse_opt_comma_list, [TRACKED],
        "only allow the listed language features to be enabled in code (comma separated)"),
    dump_mir: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "dump MIR state to file.
        `val` is used to select which passes and functions to dump."),
    join_lines: bool = (false, parse_bool, [TRACKED],
        "join \
         continued lines"),
    help: bool = (false, parse_no_value, [UNTRACKED], "Print unstable compiler options"),
    // tidy-alphabetical-end
}
"#;

        let features = parse_compiler_flags(options_rs, Path::new("options.rs"));

        assert!(features.contains_key("allow_features"));
        assert!(features.contains_key("dump_mir"));
        assert!(features.contains_key("join_lines"));
        assert!(!features.contains_key("help"));

        assert_eq!(
            features["dump_mir"].description.as_deref(),
            Some(
                "dump MIR state to file.\n        `val` is used to select which passes and functions to dump."
            ),
        );
        assert_eq!(features["join_lines"].description.as_deref(), Some("join continued lines"),);
        assert_eq!(features["allow_features"].file, PathBuf::from("options.rs"));
        assert_eq!(features["allow_features"].line, 7);
    }

    #[test]
    fn parse_one_entry_skips_help_description_and_advances() {
        let section = r#"
help: bool = (false, parse_no_value, [UNTRACKED], "Print unstable compiler options"),
join_lines: bool = (false, parse_bool, [TRACKED], "join \
    continued lines"),
"#;
        let section = section.trim_start();

        let help_entry = parse_one_entry(section, 0);
        assert_eq!(help_entry.name, "help");
        assert!(help_entry.description.is_none());

        let mut next_idx = help_entry.next_idx;
        skip_ws_comments_and_attrs(section, &mut next_idx);
        let next_entry = parse_one_entry(section, next_idx);

        assert_eq!(next_entry.name, "join_lines");
        assert_eq!(next_entry.description.as_deref(), Some("join continued lines"),);
    }

    #[test]
    fn parse_one_entry_accepts_optional_trailing_metadata() {
        let entry = r#"
deprecated_flag: bool = (false, parse_no_value, [UNTRACKED], "deprecated flag",
    is_deprecated_and_do_nothing: true),
"#;
        let entry = entry.trim_start();

        let parsed = parse_one_entry(entry, 0);
        assert_eq!(parsed.name, "deprecated_flag");
        assert_eq!(parsed.description.as_deref(), Some("deprecated flag"));
    }

    #[test]
    fn parses_real_unstable_options_file() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let options_path = manifest_dir.join("../../../compiler/rustc_session/src/options.rs");
        let options_rs = std::fs::read_to_string(&options_path).unwrap();
        let features = parse_compiler_flags(&options_rs, &options_path);

        assert!(features.contains_key("allow_features"));
        assert!(features.contains_key("dump_mir"));
        assert!(features.contains_key("unstable_options"));
        assert!(!features.contains_key("help"));
        assert!(features["dump_mir"].line > 0);
        assert!(features["dump_mir"].description.as_deref().unwrap().starts_with("dump MIR state"));
    }
}
