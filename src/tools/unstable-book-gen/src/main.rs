//! Auto-generate stub docs for the unstable book

use std::collections::BTreeSet;
use std::env;
use std::fs::{self, write};
use std::path::Path;

use proc_macro2::{Span, TokenStream, TokenTree};
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Ident, Item, LitStr, Token, parenthesized};
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
    let options_path = compiler_path.join("rustc_session/src/options/unstable.rs");
    let options_rs = t!(fs::read_to_string(&options_path), options_path);
    parse_compiler_flags(&options_rs, &options_path)
}

const DESCRIPTION_FIELD: usize = 3;
const REQUIRED_FIELDS: usize = 4;
const OPTIONAL_FIELDS: usize = 5;

struct ParsedOptionEntry {
    name: String,
    line: usize,
    description: String,
}

struct UnstableOptionsInput {
    struct_name: Ident,
    entries: Vec<ParsedOptionEntry>,
}

impl Parse for ParsedOptionEntry {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let _attrs = input.call(Attribute::parse_outer)?;

        let name: Ident = input.parse()?;
        let line = name.span().start().line;
        input.parse::<Token![:]>()?;
        let _ty: syn::Type = input.parse()?;
        input.parse::<Token![=]>()?;

        let tuple_content;
        parenthesized!(tuple_content in input);
        let tuple_tokens: TokenStream = tuple_content.parse()?;
        let tuple_fields = split_tuple_fields(tuple_tokens);

        if !matches!(tuple_fields.len(), REQUIRED_FIELDS | OPTIONAL_FIELDS) {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "unexpected field count for option `{name}`: expected {REQUIRED_FIELDS} or {OPTIONAL_FIELDS}, found {}",
                    tuple_fields.len()
                ),
            ));
        }

        if tuple_fields.len() == OPTIONAL_FIELDS
            && !is_deprecated_marker_field(&tuple_fields[REQUIRED_FIELDS])
        {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "unexpected trailing field in option `{name}`: expected `is_deprecated_and_do_nothing: ...`"
                ),
            ));
        }

        let description = parse_description_field(&tuple_fields[DESCRIPTION_FIELD], &name)?;
        Ok(Self { name: name.to_string(), line, description })
    }
}

impl Parse for UnstableOptionsInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let struct_name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let _tmod_enum_name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let _stat_name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let _opt_module_name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let _prefix: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;
        let _output_name: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;

        let entries =
            syn::punctuated::Punctuated::<ParsedOptionEntry, Token![,]>::parse_terminated(input)?
                .into_iter()
                .collect();

        Ok(Self { struct_name, entries })
    }
}

fn parse_compiler_flags(options_rs: &str, options_path: &Path) -> Features {
    let options_input = parse_unstable_options_macro(options_rs).unwrap_or_else(|error| {
        panic!("failed to parse unstable options from `{}`: {error}", options_path.display())
    });

    let mut features = Features::new();
    for entry in options_input.entries {
        if entry.name == "help" {
            continue;
        }

        features.insert(
            entry.name,
            Feature {
                level: Status::Unstable,
                since: None,
                has_gate_test: false,
                tracking_issue: None,
                file: options_path.to_path_buf(),
                line: entry.line,
                description: Some(entry.description),
            },
        );
    }

    features
}

fn parse_unstable_options_macro(source: &str) -> syn::Result<UnstableOptionsInput> {
    let ast = syn::parse_file(source)?;

    for item in ast.items {
        let Item::Macro(item_macro) = item else {
            continue;
        };

        if !item_macro.mac.path.is_ident("options") {
            continue;
        }

        let parsed = syn::parse2::<UnstableOptionsInput>(item_macro.mac.tokens)?;
        if parsed.struct_name == "UnstableOptions" {
            return Ok(parsed);
        }
    }

    Err(syn::Error::new(
        Span::call_site(),
        "could not find `options!` invocation for `UnstableOptions`",
    ))
}

fn parse_description_field(field: &TokenStream, option_name: &Ident) -> syn::Result<String> {
    let lit = syn::parse2::<LitStr>(field.clone()).map_err(|_| {
        syn::Error::new_spanned(
            field.clone(),
            format!("expected description string literal in option `{option_name}`"),
        )
    })?;
    Ok(lit.value())
}

fn split_tuple_fields(tuple_tokens: TokenStream) -> Vec<TokenStream> {
    let mut fields = Vec::new();
    let mut current = TokenStream::new();

    for token in tuple_tokens {
        if let TokenTree::Punct(punct) = &token {
            if punct.as_char() == ',' {
                fields.push(current);
                current = TokenStream::new();
                continue;
            }
        }
        current.extend([token]);
    }
    fields.push(current);

    while matches!(fields.last(), Some(field) if field.is_empty()) {
        fields.pop();
    }

    fields
}

fn is_deprecated_marker_field(field: &TokenStream) -> bool {
    let mut tokens = field.clone().into_iter();
    let Some(TokenTree::Ident(name)) = tokens.next() else {
        return false;
    };
    let Some(TokenTree::Punct(colon)) = tokens.next() else {
        return false;
    };
    name == "is_deprecated_and_do_nothing" && colon.as_char() == ':'
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
mod tests;
