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
use rustc_span::edition::Edition;
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::sym;
use rustc_span::{FileName, Span, DUMMY_SP};

use super::GlobalTestOptions;

pub(crate) struct DocTest {
    pub(crate) supports_color: bool,
    pub(crate) already_has_extern_crate: bool,
    pub(crate) main_fn_span: Option<Span>,
    pub(crate) crate_attrs: String,
    pub(crate) crates: String,
    pub(crate) everything_else: String,
    pub(crate) test_id: Option<String>,
}

impl DocTest {
    pub(crate) fn new(
        source: &str,
        crate_name: Option<&str>,
        edition: Edition,
        // If `test_id` is `None`, it means we're generating code for a code example "run" link.
        test_id: Option<String>,
    ) -> Self {
        let (crate_attrs, everything_else, crates) = partition_source(source, edition);
        let mut supports_color = false;

        // Uses librustc_ast to parse the doctest and find if there's a main fn and the extern
        // crate already is included.
        let Ok((main_fn_span, already_has_extern_crate)) =
            check_for_main_and_extern_crate(crate_name, source, edition, &mut supports_color)
        else {
            // If the parser panicked due to a fatal error, pass the test code through unchanged.
            // The error will be reported during compilation.
            return DocTest {
                supports_color: false,
                main_fn_span: None,
                crate_attrs,
                crates,
                everything_else,
                already_has_extern_crate: false,
                test_id,
            };
        };
        Self {
            supports_color,
            main_fn_span,
            crate_attrs,
            crates,
            everything_else,
            already_has_extern_crate,
            test_id,
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
        if dont_insert_main || self.main_fn_span.is_some() || prog.contains("![no_std]") {
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

fn check_for_main_and_extern_crate(
    crate_name: Option<&str>,
    source: &str,
    edition: Edition,
    supports_color: &mut bool,
) -> Result<(Option<Span>, bool), FatalError> {
    let result = rustc_driver::catch_fatal_errors(|| {
        rustc_span::create_session_if_not_set_then(edition, |_| {
            use rustc_errors::emitter::{Emitter, HumanEmitter};
            use rustc_errors::DiagCtxt;
            use rustc_parse::parser::ForceCollect;
            use rustc_span::source_map::FilePathMapping;

            let filename = FileName::anon_source_code(source);

            // Any errors in parsing should also appear when the doctest is compiled for real, so just
            // send all the errors that librustc_ast emits directly into a `Sink` instead of stderr.
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let fallback_bundle = rustc_errors::fallback_fluent_bundle(
                rustc_driver::DEFAULT_LOCALE_RESOURCES.to_vec(),
                false,
            );
            *supports_color =
                HumanEmitter::new(stderr_destination(ColorConfig::Auto), fallback_bundle.clone())
                    .supports_color();

            let emitter = HumanEmitter::new(Box::new(io::sink()), fallback_bundle);

            // FIXME(misdreavus): pass `-Z treat-err-as-bug` to the doctest parser
            let dcx = DiagCtxt::new(Box::new(emitter)).disable_warnings();
            let psess = ParseSess::with_dcx(dcx, sm);

            let mut found_main = None;
            let mut found_extern_crate = crate_name.is_none();
            let mut found_macro = false;

            let mut parser = match new_parser_from_source_str(&psess, filename, source.to_owned()) {
                Ok(p) => p,
                Err(errs) => {
                    errs.into_iter().for_each(|err| err.cancel());
                    return (found_main, found_extern_crate, found_macro);
                }
            };

            loop {
                match parser.parse_item(ForceCollect::No) {
                    Ok(Some(item)) => {
                        if found_main.is_none()
                            && let ast::ItemKind::Fn(..) = item.kind
                            && item.ident.name == sym::main
                        {
                            found_main = Some(item.span);
                        }

                        if !found_extern_crate
                            && let ast::ItemKind::ExternCrate(original) = item.kind
                        {
                            // This code will never be reached if `crate_name` is none because
                            // `found_extern_crate` is initialized to `true` if it is none.
                            let crate_name = crate_name.unwrap();

                            match original {
                                Some(name) => found_extern_crate = name.as_str() == crate_name,
                                None => found_extern_crate = item.ident.as_str() == crate_name,
                            }
                        }

                        if !found_macro && let ast::ItemKind::MacCall(..) = item.kind {
                            found_macro = true;
                        }

                        if found_main.is_some() && found_extern_crate {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        e.cancel();
                        break;
                    }
                }

                // The supplied item is only used for diagnostics,
                // which are swallowed here anyway.
                parser.maybe_consume_incorrect_semicolon(None);
            }

            // Reset errors so that they won't be reported as compiler bugs when dropping the
            // dcx. Any errors in the tests will be reported when the test file is compiled,
            // Note that we still need to cancel the errors above otherwise `Diag` will panic on
            // drop.
            psess.dcx().reset_err_count();

            (found_main, found_extern_crate, found_macro)
        })
    });
    let (mut main_fn_span, already_has_extern_crate, found_macro) = result?;

    // If a doctest's `fn main` is being masked by a wrapper macro, the parsing loop above won't
    // see it. In that case, run the old text-based scan to see if they at least have a main
    // function written inside a macro invocation. See
    // https://github.com/rust-lang/rust/issues/56898
    if found_macro
        && main_fn_span.is_none()
        && source
            .lines()
            .map(|line| {
                let comment = line.find("//");
                if let Some(comment_begins) = comment { &line[0..comment_begins] } else { line }
            })
            .any(|code| code.contains("fn main"))
    {
        main_fn_span = Some(DUMMY_SP);
    }

    Ok((main_fn_span, already_has_extern_crate))
}

fn check_if_attr_is_complete(source: &str, edition: Edition) -> bool {
    if source.is_empty() {
        // Empty content so nothing to check in here...
        return true;
    }
    rustc_driver::catch_fatal_errors(|| {
        rustc_span::create_session_if_not_set_then(edition, |_| {
            use rustc_errors::emitter::HumanEmitter;
            use rustc_errors::DiagCtxt;
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
                    return false;
                }
            };
            // If a parsing error happened, it's very likely that the attribute is incomplete.
            if let Err(e) = parser.parse_attribute(InnerAttrPolicy::Permitted) {
                e.cancel();
                return false;
            }
            true
        })
    })
    .unwrap_or(false)
}

fn partition_source(s: &str, edition: Edition) -> (String, String, String) {
    #[derive(Copy, Clone, PartialEq)]
    enum PartitionState {
        Attrs,
        Crates,
        Other,
    }
    let mut state = PartitionState::Attrs;
    let mut before = String::new();
    let mut crates = String::new();
    let mut after = String::new();

    let mut mod_attr_pending = String::new();

    for line in s.lines() {
        let trimline = line.trim();

        // FIXME(misdreavus): if a doc comment is placed on an extern crate statement, it will be
        // shunted into "everything else"
        match state {
            PartitionState::Attrs => {
                state = if trimline.starts_with("#![") {
                    if !check_if_attr_is_complete(line, edition) {
                        mod_attr_pending = line.to_owned();
                    } else {
                        mod_attr_pending.clear();
                    }
                    PartitionState::Attrs
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
                        if !trimline.is_empty()
                            && check_if_attr_is_complete(&mod_attr_pending, edition)
                        {
                            // If it's complete, then we can clear the pending content.
                            mod_attr_pending.clear();
                        }
                        // In any case, this is considered as `PartitionState::Attrs` so it's
                        // prepended before rustdoc's inserts.
                        PartitionState::Attrs
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
                before.push_str(line);
                before.push('\n');
            }
            PartitionState::Crates => {
                crates.push_str(line);
                crates.push('\n');
            }
            PartitionState::Other => {
                after.push_str(line);
                after.push('\n');
            }
        }
    }

    debug!("before:\n{before}");
    debug!("crates:\n{crates}");
    debug!("after:\n{after}");

    (before, after, crates)
}
