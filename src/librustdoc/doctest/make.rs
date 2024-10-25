//! Logic for transforming the raw code given by the user into something actually
//! runnable, e.g. by adding a `main` function if it doesn't already exist.

use std::io;

use rustc_ast as ast;
use rustc_data_structures::sync::Lrc;
use rustc_errors::emitter::stderr_destination;
use rustc_errors::{ColorConfig, FatalError};
use rustc_parse::new_parser_from_source_str;
use rustc_parse::parser::attr::InnerAttrPolicy;
use rustc_session::parse::ParseSess;
use rustc_span::FileName;
use rustc_span::edition::Edition;
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::sym;
use tracing::debug;

use super::GlobalTestOptions;
use crate::html::markdown::LangString;

/// This struct contains information about the doctest itself which is then used to generate
/// doctest source code appropriately.
pub(crate) struct DocTestBuilder {
    pub(crate) supports_color: bool,
    pub(crate) already_has_extern_crate: bool,
    pub(crate) has_main_fn: bool,
    pub(crate) crate_attrs: String,
    /// If this is a merged doctest, it will be put into `everything_else`, otherwise it will
    /// put into `crate_attrs`.
    pub(crate) maybe_crate_attrs: String,
    pub(crate) crates: String,
    pub(crate) everything_else: String,
    pub(crate) test_id: Option<String>,
    pub(crate) failed_ast: bool,
    pub(crate) can_be_merged: bool,
}

impl DocTestBuilder {
    pub(crate) fn new(
        source: &str,
        crate_name: Option<&str>,
        edition: Edition,
        can_merge_doctests: bool,
        // If `test_id` is `None`, it means we're generating code for a code example "run" link.
        test_id: Option<String>,
        lang_str: Option<&LangString>,
    ) -> Self {
        let can_merge_doctests = can_merge_doctests
            && lang_str.is_some_and(|lang_str| {
                !lang_str.compile_fail && !lang_str.test_harness && !lang_str.standalone_crate
            });

        let SourceInfo { crate_attrs, maybe_crate_attrs, crates, everything_else } =
            partition_source(source, edition);

        // Uses librustc_ast to parse the doctest and find if there's a main fn and the extern
        // crate already is included.
        let Ok((
            ParseSourceInfo {
                has_main_fn,
                found_extern_crate,
                supports_color,
                has_global_allocator,
                has_macro_def,
                ..
            },
            failed_ast,
        )) = check_for_main_and_extern_crate(
            crate_name,
            source,
            &everything_else,
            &crates,
            edition,
            can_merge_doctests,
        )
        else {
            // If the parser panicked due to a fatal error, pass the test code through unchanged.
            // The error will be reported during compilation.
            return Self {
                supports_color: false,
                has_main_fn: false,
                crate_attrs,
                maybe_crate_attrs,
                crates,
                everything_else,
                already_has_extern_crate: false,
                test_id,
                failed_ast: true,
                can_be_merged: false,
            };
        };
        // If the AST returned an error, we don't want this doctest to be merged with the
        // others. Same if it contains `#[feature]` or `#[no_std]`.
        let can_be_merged = can_merge_doctests
            && !failed_ast
            && !has_global_allocator
            && crate_attrs.is_empty()
            // If this is a merged doctest and a defined macro uses `$crate`, then the path will
            // not work, so better not put it into merged doctests.
            && !(has_macro_def && everything_else.contains("$crate"));
        Self {
            supports_color,
            has_main_fn,
            crate_attrs,
            maybe_crate_attrs,
            crates,
            everything_else,
            already_has_extern_crate: found_extern_crate,
            test_id,
            failed_ast: false,
            can_be_merged,
        }
    }

    /// Transforms a test into code that can be compiled into a Rust binary, and returns the number of
    /// lines before the test code begins.
    pub(crate) fn generate_unique_doctest(
        &self,
        test_code: &str,
        dont_insert_main: bool,
        opts: &GlobalTestOptions,
        crate_name: Option<&str>,
    ) -> (String, usize) {
        if self.failed_ast {
            // If the AST failed to compile, no need to go generate a complete doctest, the error
            // will be better this way.
            return (test_code.to_string(), 0);
        }
        let mut line_offset = 0;
        let mut prog = String::new();
        let everything_else = self.everything_else.trim();
        if opts.attrs.is_empty() {
            // If there aren't any attributes supplied by #![doc(test(attr(...)))], then allow some
            // lints that are commonly triggered in doctests. The crate-level test attributes are
            // commonly used to make tests fail in case they trigger warnings, so having this there in
            // that case may cause some tests to pass when they shouldn't have.
            prog.push_str("#![allow(unused)]\n");
            line_offset += 1;
        }

        // Next, any attributes that came from the crate root via #![doc(test(attr(...)))].
        for attr in &opts.attrs {
            prog.push_str(&format!("#![{attr}]\n"));
            line_offset += 1;
        }

        // Now push any outer attributes from the example, assuming they
        // are intended to be crate attributes.
        prog.push_str(&self.crate_attrs);
        prog.push_str(&self.maybe_crate_attrs);
        prog.push_str(&self.crates);

        // Don't inject `extern crate std` because it's already injected by the
        // compiler.
        if !self.already_has_extern_crate &&
            !opts.no_crate_inject &&
            let Some(crate_name) = crate_name &&
            crate_name != "std" &&
            // Don't inject `extern crate` if the crate is never used.
            // NOTE: this is terribly inaccurate because it doesn't actually
            // parse the source, but only has false positives, not false
            // negatives.
            test_code.contains(crate_name)
        {
            // rustdoc implicitly inserts an `extern crate` item for the own crate
            // which may be unused, so we need to allow the lint.
            prog.push_str("#[allow(unused_extern_crates)]\n");

            prog.push_str(&format!("extern crate r#{crate_name};\n"));
            line_offset += 1;
        }

        // FIXME: This code cannot yet handle no_std test cases yet
        if dont_insert_main || self.has_main_fn || prog.contains("![no_std]") {
            prog.push_str(everything_else);
        } else {
            let returns_result = everything_else.ends_with("(())");
            // Give each doctest main function a unique name.
            // This is for example needed for the tooling around `-C instrument-coverage`.
            let inner_fn_name = if let Some(ref test_id) = self.test_id {
                format!("_doctest_main_{test_id}")
            } else {
                "_inner".into()
            };
            let inner_attr = if self.test_id.is_some() { "#[allow(non_snake_case)] " } else { "" };
            let (main_pre, main_post) = if returns_result {
                (
                    format!(
                        "fn main() {{ {inner_attr}fn {inner_fn_name}() -> Result<(), impl core::fmt::Debug> {{\n",
                    ),
                    format!("\n}} {inner_fn_name}().unwrap() }}"),
                )
            } else if self.test_id.is_some() {
                (
                    format!("fn main() {{ {inner_attr}fn {inner_fn_name}() {{\n",),
                    format!("\n}} {inner_fn_name}() }}"),
                )
            } else {
                ("fn main() {\n".into(), "\n}".into())
            };
            // Note on newlines: We insert a line/newline *before*, and *after*
            // the doctest and adjust the `line_offset` accordingly.
            // In the case of `-C instrument-coverage`, this means that the generated
            // inner `main` function spans from the doctest opening codeblock to the
            // closing one. For example
            // /// ``` <- start of the inner main
            // /// <- code under doctest
            // /// ``` <- end of the inner main
            line_offset += 1;

            prog.push_str(&main_pre);

            // add extra 4 spaces for each line to offset the code block
            if opts.insert_indent_space {
                prog.push_str(
                    &everything_else
                        .lines()
                        .map(|line| format!("    {}", line))
                        .collect::<Vec<String>>()
                        .join("\n"),
                );
            } else {
                prog.push_str(everything_else);
            };
            prog.push_str(&main_post);
        }

        debug!("final doctest:\n{prog}");

        (prog, line_offset)
    }
}

#[derive(PartialEq, Eq, Debug)]
enum ParsingResult {
    Failed,
    AstError,
    Ok,
}

fn cancel_error_count(psess: &ParseSess) {
    // Reset errors so that they won't be reported as compiler bugs when dropping the
    // dcx. Any errors in the tests will be reported when the test file is compiled,
    // Note that we still need to cancel the errors above otherwise `Diag` will panic on
    // drop.
    psess.dcx().reset_err_count();
}

fn parse_source(
    source: String,
    info: &mut ParseSourceInfo,
    crate_name: &Option<&str>,
) -> ParsingResult {
    use rustc_errors::DiagCtxt;
    use rustc_errors::emitter::{Emitter, HumanEmitter};
    use rustc_parse::parser::ForceCollect;
    use rustc_span::source_map::FilePathMapping;

    let filename = FileName::anon_source_code(&source);

    // Any errors in parsing should also appear when the doctest is compiled for real, so just
    // send all the errors that librustc_ast emits directly into a `Sink` instead of stderr.
    let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
    let fallback_bundle = rustc_errors::fallback_fluent_bundle(
        rustc_driver::DEFAULT_LOCALE_RESOURCES.to_vec(),
        false,
    );
    info.supports_color =
        HumanEmitter::new(stderr_destination(ColorConfig::Auto), fallback_bundle.clone())
            .supports_color();

    let emitter = HumanEmitter::new(Box::new(io::sink()), fallback_bundle);

    // FIXME(misdreavus): pass `-Z treat-err-as-bug` to the doctest parser
    let dcx = DiagCtxt::new(Box::new(emitter)).disable_warnings();
    let psess = ParseSess::with_dcx(dcx, sm);

    let mut parser = match new_parser_from_source_str(&psess, filename, source) {
        Ok(p) => p,
        Err(errs) => {
            errs.into_iter().for_each(|err| err.cancel());
            cancel_error_count(&psess);
            return ParsingResult::Failed;
        }
    };
    let mut parsing_result = ParsingResult::Ok;

    // Recurse through functions body. It is necessary because the doctest source code is
    // wrapped in a function to limit the number of AST errors. If we don't recurse into
    // functions, we would thing all top-level items (so basically nothing).
    fn check_item(
        item: &ast::Item,
        info: &mut ParseSourceInfo,
        crate_name: &Option<&str>,
        is_top_level: bool,
    ) {
        if !info.has_global_allocator
            && item.attrs.iter().any(|attr| attr.name_or_empty() == sym::global_allocator)
        {
            info.has_global_allocator = true;
        }
        match item.kind {
            ast::ItemKind::Fn(ref fn_item) if !info.has_main_fn => {
                if item.ident.name == sym::main && is_top_level {
                    info.has_main_fn = true;
                }
                if let Some(ref body) = fn_item.body {
                    for stmt in &body.stmts {
                        match stmt.kind {
                            ast::StmtKind::Item(ref item) => {
                                check_item(item, info, crate_name, false)
                            }
                            ast::StmtKind::MacCall(..) => info.found_macro = true,
                            _ => {}
                        }
                    }
                }
            }
            ast::ItemKind::ExternCrate(original) => {
                if !info.found_extern_crate
                    && let Some(crate_name) = crate_name
                {
                    info.found_extern_crate = match original {
                        Some(name) => name.as_str() == *crate_name,
                        None => item.ident.as_str() == *crate_name,
                    };
                }
            }
            ast::ItemKind::MacCall(..) => info.found_macro = true,
            ast::ItemKind::MacroDef(..) => info.has_macro_def = true,
            _ => {}
        }
    }

    loop {
        match parser.parse_item(ForceCollect::No) {
            Ok(Some(item)) => {
                check_item(&item, info, crate_name, true);

                if info.has_main_fn && info.found_extern_crate {
                    break;
                }
            }
            Ok(None) => break,
            Err(e) => {
                parsing_result = ParsingResult::AstError;
                e.cancel();
                break;
            }
        }

        // The supplied item is only used for diagnostics,
        // which are swallowed here anyway.
        parser.maybe_consume_incorrect_semicolon(None);
    }

    cancel_error_count(&psess);
    parsing_result
}

#[derive(Default)]
struct ParseSourceInfo {
    has_main_fn: bool,
    found_extern_crate: bool,
    found_macro: bool,
    supports_color: bool,
    has_global_allocator: bool,
    has_macro_def: bool,
}

fn check_for_main_and_extern_crate(
    crate_name: Option<&str>,
    original_source_code: &str,
    everything_else: &str,
    crates: &str,
    edition: Edition,
    can_merge_doctests: bool,
) -> Result<(ParseSourceInfo, bool), FatalError> {
    let result = rustc_driver::catch_fatal_errors(|| {
        rustc_span::create_session_if_not_set_then(edition, |_| {
            let mut info =
                ParseSourceInfo { found_extern_crate: crate_name.is_none(), ..Default::default() };

            let mut parsing_result =
                parse_source(format!("{crates}{everything_else}"), &mut info, &crate_name);
            // No need to double-check this if the "merged doctests" feature isn't enabled (so
            // before the 2024 edition).
            if can_merge_doctests && parsing_result != ParsingResult::Ok {
                // If we found an AST error, we want to ensure it's because of an expression being
                // used outside of a function.
                //
                // To do so, we wrap in a function in order to make sure that the doctest AST is
                // correct. For example, if your doctest is `foo::bar()`, if we don't wrap it in a
                // block, it would emit an AST error, which would be problematic for us since we
                // want to filter out such errors which aren't "real" errors.
                //
                // The end goal is to be able to merge as many doctests as possible as one for much
                // faster doctests run time.
                parsing_result = parse_source(
                    format!("{crates}\nfn __doctest_wrap(){{{everything_else}\n}}"),
                    &mut info,
                    &crate_name,
                );
            }

            (info, parsing_result)
        })
    });
    let (mut info, parsing_result) = match result {
        Err(..) | Ok((_, ParsingResult::Failed)) => return Err(FatalError),
        Ok((info, parsing_result)) => (info, parsing_result),
    };

    // If a doctest's `fn main` is being masked by a wrapper macro, the parsing loop above won't
    // see it. In that case, run the old text-based scan to see if they at least have a main
    // function written inside a macro invocation. See
    // https://github.com/rust-lang/rust/issues/56898
    if info.found_macro
        && !info.has_main_fn
        && original_source_code
            .lines()
            .map(|line| {
                let comment = line.find("//");
                if let Some(comment_begins) = comment { &line[0..comment_begins] } else { line }
            })
            .any(|code| code.contains("fn main"))
    {
        info.has_main_fn = true;
    }

    Ok((info, parsing_result != ParsingResult::Ok))
}

enum AttrKind {
    CrateAttr,
    Attr,
}

/// Returns `Some` if the attribute is complete and `Some(true)` if it is an attribute that can be
/// placed at the crate root.
fn check_if_attr_is_complete(source: &str, edition: Edition) -> Option<AttrKind> {
    if source.is_empty() {
        // Empty content so nothing to check in here...
        return None;
    }
    let not_crate_attrs = [sym::forbid, sym::allow, sym::warn, sym::deny];

    rustc_driver::catch_fatal_errors(|| {
        rustc_span::create_session_if_not_set_then(edition, |_| {
            use rustc_errors::DiagCtxt;
            use rustc_errors::emitter::HumanEmitter;
            use rustc_span::source_map::FilePathMapping;

            let filename = FileName::anon_source_code(source);
            // Any errors in parsing should also appear when the doctest is compiled for real, so just
            // send all the errors that librustc_ast emits directly into a `Sink` instead of stderr.
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let fallback_bundle = rustc_errors::fallback_fluent_bundle(
                rustc_driver::DEFAULT_LOCALE_RESOURCES.to_vec(),
                false,
            );

            let emitter = HumanEmitter::new(Box::new(io::sink()), fallback_bundle);

            let dcx = DiagCtxt::new(Box::new(emitter)).disable_warnings();
            let psess = ParseSess::with_dcx(dcx, sm);
            let mut parser = match new_parser_from_source_str(&psess, filename, source.to_owned()) {
                Ok(p) => p,
                Err(errs) => {
                    errs.into_iter().for_each(|err| err.cancel());
                    // If there is an unclosed delimiter, an error will be returned by the
                    // tokentrees.
                    return None;
                }
            };
            // If a parsing error happened, it's very likely that the attribute is incomplete.
            let ret = match parser.parse_attribute(InnerAttrPolicy::Permitted) {
                Ok(attr) => {
                    let attr_name = attr.name_or_empty();

                    if not_crate_attrs.contains(&attr_name) {
                        // There is one exception to these attributes:
                        // `#![allow(internal_features)]`. If this attribute is used, we need to
                        // consider it only as a crate-level attribute.
                        if attr_name == sym::allow
                            && let Some(list) = attr.meta_item_list()
                            && list.iter().any(|sub_attr| {
                                sub_attr.name_or_empty().as_str() == "internal_features"
                            })
                        {
                            Some(AttrKind::CrateAttr)
                        } else {
                            Some(AttrKind::Attr)
                        }
                    } else {
                        Some(AttrKind::CrateAttr)
                    }
                }
                Err(e) => {
                    e.cancel();
                    None
                }
            };
            ret
        })
    })
    .unwrap_or(None)
}

fn handle_attr(mod_attr_pending: &mut String, source_info: &mut SourceInfo, edition: Edition) {
    if let Some(attr_kind) = check_if_attr_is_complete(mod_attr_pending, edition) {
        let push_to = match attr_kind {
            AttrKind::CrateAttr => &mut source_info.crate_attrs,
            AttrKind::Attr => &mut source_info.maybe_crate_attrs,
        };
        push_to.push_str(mod_attr_pending);
        push_to.push('\n');
        // If it's complete, then we can clear the pending content.
        mod_attr_pending.clear();
    } else if mod_attr_pending.ends_with('\\') {
        mod_attr_pending.push('n');
    }
}

#[derive(Default)]
struct SourceInfo {
    crate_attrs: String,
    maybe_crate_attrs: String,
    crates: String,
    everything_else: String,
}

fn partition_source(s: &str, edition: Edition) -> SourceInfo {
    #[derive(Copy, Clone, PartialEq)]
    enum PartitionState {
        Attrs,
        Crates,
        Other,
    }
    let mut source_info = SourceInfo::default();
    let mut state = PartitionState::Attrs;
    let mut mod_attr_pending = String::new();

    for line in s.lines() {
        let trimline = line.trim();

        // FIXME(misdreavus): if a doc comment is placed on an extern crate statement, it will be
        // shunted into "everything else"
        match state {
            PartitionState::Attrs => {
                state = if trimline.starts_with("#![") {
                    mod_attr_pending = line.to_owned();
                    handle_attr(&mut mod_attr_pending, &mut source_info, edition);
                    continue;
                } else if trimline.chars().all(|c| c.is_whitespace())
                    || (trimline.starts_with("//") && !trimline.starts_with("///"))
                {
                    PartitionState::Attrs
                } else if trimline.starts_with("extern crate")
                    || trimline.starts_with("#[macro_use] extern crate")
                {
                    PartitionState::Crates
                } else {
                    // First we check if the previous attribute was "complete"...
                    if !mod_attr_pending.is_empty() {
                        // If not, then we append the new line into the pending attribute to check
                        // if this time it's complete...
                        mod_attr_pending.push_str(line);
                        if !trimline.is_empty() {
                            handle_attr(&mut mod_attr_pending, &mut source_info, edition);
                        }
                        continue;
                    } else {
                        PartitionState::Other
                    }
                };
            }
            PartitionState::Crates => {
                state = if trimline.starts_with("extern crate")
                    || trimline.starts_with("#[macro_use] extern crate")
                    || trimline.chars().all(|c| c.is_whitespace())
                    || (trimline.starts_with("//") && !trimline.starts_with("///"))
                {
                    PartitionState::Crates
                } else {
                    PartitionState::Other
                };
            }
            PartitionState::Other => {}
        }

        match state {
            PartitionState::Attrs => {
                source_info.crate_attrs.push_str(line);
                source_info.crate_attrs.push('\n');
            }
            PartitionState::Crates => {
                source_info.crates.push_str(line);
                source_info.crates.push('\n');
            }
            PartitionState::Other => {
                source_info.everything_else.push_str(line);
                source_info.everything_else.push('\n');
            }
        }
    }

    source_info.everything_else = source_info.everything_else.trim().to_string();

    debug!("crate_attrs:\n{}{}", source_info.crate_attrs, source_info.maybe_crate_attrs);
    debug!("crates:\n{}", source_info.crates);
    debug!("after:\n{}", source_info.everything_else);

    source_info
}
