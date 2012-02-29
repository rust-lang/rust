import std::getopts;

export output_format::{};
export output_style::{};
export config;
export parse_config;
export usage;

#[doc = "The type of document to output"]
enum output_format {
    #[doc = "Markdown"]
    markdown,
    #[doc = "HTML, via markdown and pandoc"]
    pandoc_html
}

#[doc = "How to organize the output"]
enum output_style {
    #[doc = "All in a single document"]
    doc_per_crate,
    #[doc = "Each module in its own document"]
    doc_per_mod
}

#[doc = "The configuration for a rustdoc session"]
type config = {
    input_crate: str,
    output_dir: str,
    output_format: output_format,
    output_style: output_style,
    pandoc_cmd: option<str>
};

fn opt_output_dir() -> str { "output-dir" }
fn opt_output_format() -> str { "output-format" }
fn opt_output_style() -> str { "output-style" }
fn opt_pandoc_cmd() -> str { "pandoc-cmd" }
fn opt_help() -> str { "h" }

fn opts() -> [(getopts::opt, str)] {
    [
        (getopts::optopt(opt_output_dir()),
         "--output-dir <val>     put documents here"),
        (getopts::optopt(opt_output_format()),
         "--output-format <val>  either 'markdown' or 'html'"),
        (getopts::optopt(opt_output_style()),
         "--output-style <val>   either 'doc-per-crate' or 'doc-per-mod'"),
        (getopts::optopt(opt_pandoc_cmd()),
         "--pandoc-cmd <val>     the command for running pandoc"),
        (getopts::optflag(opt_help()),
         "-h                     print help")
    ]
}

fn usage() {
    import std::io::println;

    println("Usage: rustdoc [options] <cratefile>\n");
    println("Options:\n");
    for opt in opts() {
        println(#fmt("    %s", tuple::second(opt)));
    }
    println("");
}

fn default_config(input_crate: str) -> config {
    {
        input_crate: input_crate,
        output_dir: ".",
        output_format: pandoc_html,
        output_style: doc_per_mod,
        pandoc_cmd: none
    }
}

fn parse_config(args: [str]) -> result::t<config, str> {
    let args = vec::tail(args);
    let opts = tuple::first(vec::unzip(opts()));
    alt getopts::getopts(args, opts) {
        result::ok(match) {
            if vec::len(match.free) == 1u {
                let input_crate = vec::head(match.free);
                config_from_opts(input_crate, match)
            } else if vec::is_empty(match.free) {
                result::err("no crates specified")
            } else {
                result::err("multiple crates specified")
            }
        }
        result::err(f) {
            result::err(getopts::fail_str(f))
        }
    }
}

fn config_from_opts(
    input_crate: str,
    match: getopts::match
) -> result::t<config, str> {

    let config = default_config(input_crate);
    let result = result::ok(config);
    let result = result::chain(result) {|config|
        let output_dir = getopts::opt_maybe_str(match, opt_output_dir());
        result::ok({
            output_dir: option::from_maybe(config.output_dir, output_dir)
            with config
        })
    };
    let result = result::chain(result) {|config|
        let output_format = getopts::opt_maybe_str(
            match, opt_output_format());
        option::maybe(result::ok(config), output_format) {|output_format|
            result::chain(parse_output_format(output_format)) {|output_format|
                result::ok({
                    output_format: output_format
                    with config
                })
            }
        }
    };
    let result = result::chain(result) {|config|
        let output_style = getopts::opt_maybe_str(match, opt_output_style());
        option::maybe(result::ok(config), output_style) {|output_style|
            result::chain(parse_output_style(output_style)) {|output_style|
                result::ok({
                    output_style: output_style
                    with config
                })
            }
        }
    };
    let result = result::chain(result) {|config|
        let pandoc_cmd = getopts::opt_maybe_str(match, opt_pandoc_cmd());
        let pandoc_cmd = maybe_find_pandoc(config, pandoc_cmd);
        result::chain(pandoc_cmd) {|pandoc_cmd|
            result::ok({
                pandoc_cmd: pandoc_cmd
                with config
            })
        }
    };
    ret result;
}

fn parse_output_format(output_format: str) -> result::t<output_format, str> {
    alt output_format {
      "markdown" { result::ok(markdown) }
      "html" { result::ok(pandoc_html) }
      _ { result::err(#fmt("unknown output format '%s'", output_format)) }
    }
}

fn parse_output_style(output_style: str) -> result::t<output_style, str> {
    alt output_style {
      "doc-per-crate" { result::ok(doc_per_crate) }
      "doc-per-mod" { result::ok(doc_per_mod) }
      _ { result::err(#fmt("unknown output style '%s'", output_style)) }
    }
}

fn maybe_find_pandoc(
    _config: config,
    maybe_pandoc_cmd: option<str>
) -> result::t<option<str>, str> {
    // FIXME: When we actually need pandoc then look for it first
    // on the path, then in cabal; test to make sure pandoc works
    alt maybe_pandoc_cmd {
      some(pandoc_cmd) { result::ok(some(pandoc_cmd)) }
      none { result::ok(some("pandoc")) }
    }
}

#[test]
fn should_error_with_no_crates() {
    let config = parse_config(["rustdoc"]);
    assert result::get_err(config) == "no crates specified";
}

#[test]
fn should_error_with_multiple_crates() {
    let config = parse_config(["rustdoc", "crate1.rc", "crate2.rc"]);
    assert result::get_err(config) == "multiple crates specified";
}

#[test]
fn should_set_output_dir_to_cwd_if_not_provided() {
    let config = parse_config(["rustdoc", "crate.rc"]);
    assert result::get(config).output_dir == ".";
}

#[test]
fn should_set_output_dir_if_provided() {
    let config = parse_config([
        "rustdoc", "crate.rc", "--output-dir", "snuggles"
    ]);
    assert result::get(config).output_dir == "snuggles";
}

#[test]
fn should_set_output_format_to_pandoc_html_if_not_provided() {
    let config = parse_config(["rustdoc", "crate.rc"]);
    assert result::get(config).output_format == pandoc_html;
}

#[test]
fn should_set_output_format_to_markdown_if_requested() {
    let config = parse_config([
        "rustdoc", "crate.rc", "--output-format", "markdown"
    ]);
    assert result::get(config).output_format == markdown;
}

#[test]
fn should_set_output_format_to_pandoc_html_if_requested() {
    let config = parse_config([
        "rustdoc", "crate.rc", "--output-format", "html"
    ]);
    assert result::get(config).output_format == pandoc_html;
}

#[test]
fn should_error_on_bogus_format() {
    let config = parse_config([
        "rustdoc", "crate.rc", "--output-format", "bogus"
    ]);
    assert result::get_err(config) == "unknown output format 'bogus'";
}

#[test]
fn should_set_output_style_to_doc_per_mod_by_default() {
    let config = parse_config(["rustdoc", "crate.rc"]);
    assert result::get(config).output_style == doc_per_mod;
}

#[test]
fn should_set_output_style_to_one_doc_if_requested() {
    let config = parse_config([
        "rustdoc", "crate.rc", "--output-style", "doc-per-crate"
    ]);
    assert result::get(config).output_style == doc_per_crate;
}

#[test]
fn should_set_output_style_to_doc_per_mod_if_requested() {
    let config = parse_config([
        "rustdoc", "crate.rc", "--output-style", "doc-per-mod"
    ]);
    assert result::get(config).output_style == doc_per_mod;
}

#[test]
fn should_error_on_bogus_output_style() {
    let config = parse_config([
        "rustdoc", "crate.rc", "--output-style", "bogus"
    ]);
    assert result::get_err(config) == "unknown output style 'bogus'";
}

#[test]
fn should_set_pandoc_command_if_requested() {
    let config = parse_config([
        "rustdoc", "crate.rc", "--pandoc-cmd", "panda-bear-doc"
    ]);
    assert result::get(config).pandoc_cmd == some("panda-bear-doc");
}

#[test]
fn should_set_pandoc_command_when_using_pandoc() {
    let config = parse_config(["rustdoc", "crate.rc"]);
    assert result::get(config).pandoc_cmd == some("pandoc");
}