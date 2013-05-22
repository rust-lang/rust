// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use core::cell::Cell;
use core::os;
use core::result::Result;
use core::result;
use core::run::ProcessOutput;
use core::run;
use core::vec;
use extra::getopts;

/// The type of document to output
#[deriving(Eq)]
pub enum OutputFormat {
    /// Markdown
    pub Markdown,
    /// HTML, via markdown and pandoc
    pub PandocHtml
}

/// How to organize the output
#[deriving(Eq)]
pub enum OutputStyle {
    /// All in a single document
    pub DocPerCrate,
    /// Each module in its own document
    pub DocPerMod
}

/// The configuration for a rustdoc session
pub struct Config {
    input_crate: Path,
    output_dir: Path,
    output_format: OutputFormat,
    output_style: OutputStyle,
    pandoc_cmd: Option<~str>
}

impl Clone for Config {
    fn clone(&self) -> Config { copy *self }
}

fn opt_output_dir() -> ~str { ~"output-dir" }
fn opt_output_format() -> ~str { ~"output-format" }
fn opt_output_style() -> ~str { ~"output-style" }
fn opt_pandoc_cmd() -> ~str { ~"pandoc-cmd" }
fn opt_help() -> ~str { ~"h" }

fn opts() -> ~[(getopts::Opt, ~str)] {
    ~[
        (getopts::optopt(opt_output_dir()),
         ~"--output-dir <val>     put documents here"),
        (getopts::optopt(opt_output_format()),
         ~"--output-format <val>  either 'markdown' or 'html'"),
        (getopts::optopt(opt_output_style()),
         ~"--output-style <val>   either 'doc-per-crate' or 'doc-per-mod'"),
        (getopts::optopt(opt_pandoc_cmd()),
         ~"--pandoc-cmd <val>     the command for running pandoc"),
        (getopts::optflag(opt_help()),
         ~"-h                     print help")
    ]
}

pub fn usage() {
    use core::io::println;

    println("Usage: rustdoc [options] <cratefile>\n");
    println("Options:\n");
    for opts().each |opt| {
        println(fmt!("    %s", opt.second()));
    }
    println("");
}

pub fn default_config(input_crate: &Path) -> Config {
    Config {
        input_crate: copy *input_crate,
        output_dir: Path("."),
        output_format: PandocHtml,
        output_style: DocPerMod,
        pandoc_cmd: None
    }
}

type Process = ~fn((&str), (&[~str])) -> ProcessOutput;

pub fn mock_process_output(_prog: &str, _args: &[~str]) -> ProcessOutput {
    ProcessOutput {
        status: 0,
        output: ~[],
        error: ~[]
    }
}

pub fn process_output(prog: &str, args: &[~str]) -> ProcessOutput {
    run::process_output(prog, args)
}

pub fn parse_config(args: &[~str]) -> Result<Config, ~str> {
    parse_config_(args, process_output)
}

pub fn parse_config_(
    args: &[~str],
    process_output: Process
) -> Result<Config, ~str> {
    let args = args.tail();
    let opts = vec::unzip(opts()).first();
    match getopts::getopts(args, opts) {
        Ok(matches) => {
            if matches.free.len() == 1 {
                let input_crate = Path(*matches.free.head());
                config_from_opts(&input_crate, &matches, process_output)
            } else if matches.free.is_empty() {
                Err(~"no crates specified")
            } else {
                Err(~"multiple crates specified")
            }
        }
        Err(f) => {
            Err(getopts::fail_str(f))
        }
    }
}

fn config_from_opts(
    input_crate: &Path,
    matches: &getopts::Matches,
    process_output: Process
) -> Result<Config, ~str> {

    let config = default_config(input_crate);
    let result = result::Ok(config);
    let result = do result::chain(result) |config| {
        let output_dir = getopts::opt_maybe_str(matches, opt_output_dir());
        let output_dir = output_dir.map(|s| Path(*s));
        result::Ok(Config {
            output_dir: output_dir.get_or_default(copy config.output_dir),
            .. config
        })
    };
    let result = do result::chain(result) |config| {
        let output_format = getopts::opt_maybe_str(
            matches, opt_output_format());
        do output_format.map_default(result::Ok(copy config))
            |output_format| {
            do result::chain(parse_output_format(*output_format))
                |output_format| {

                result::Ok(Config {
                    output_format: output_format,
                    .. copy config
                })
            }
        }
    };
    let result = do result::chain(result) |config| {
        let output_style =
            getopts::opt_maybe_str(matches, opt_output_style());
        do output_style.map_default(result::Ok(copy config))
            |output_style| {
            do result::chain(parse_output_style(*output_style))
                |output_style| {
                result::Ok(Config {
                    output_style: output_style,
                    .. copy config
                })
            }
        }
    };
    let process_output = Cell::new(process_output);
    let result = do result::chain(result) |config| {
        let pandoc_cmd = getopts::opt_maybe_str(matches, opt_pandoc_cmd());
        let pandoc_cmd = maybe_find_pandoc(
            &config, pandoc_cmd, process_output.take());
        do result::chain(pandoc_cmd) |pandoc_cmd| {
            result::Ok(Config {
                pandoc_cmd: pandoc_cmd,
                .. copy config
            })
        }
    };
    return result;
}

fn parse_output_format(output_format: &str) -> Result<OutputFormat, ~str> {
    match output_format.to_str() {
      ~"markdown" => result::Ok(Markdown),
      ~"html" => result::Ok(PandocHtml),
      _ => result::Err(fmt!("unknown output format '%s'", output_format))
    }
}

fn parse_output_style(output_style: &str) -> Result<OutputStyle, ~str> {
    match output_style.to_str() {
      ~"doc-per-crate" => result::Ok(DocPerCrate),
      ~"doc-per-mod" => result::Ok(DocPerMod),
      _ => result::Err(fmt!("unknown output style '%s'", output_style))
    }
}

pub fn maybe_find_pandoc(
    config: &Config,
    maybe_pandoc_cmd: Option<~str>,
    process_output: Process
) -> Result<Option<~str>, ~str> {
    if config.output_format != PandocHtml {
        return result::Ok(maybe_pandoc_cmd);
    }

    let possible_pandocs = match maybe_pandoc_cmd {
      Some(pandoc_cmd) => ~[pandoc_cmd],
      None => {
        ~[~"pandoc"] + match os::homedir() {
          Some(dir) => {
            ~[dir.push_rel(&Path(".cabal/bin/pandoc")).to_str()]
          }
          None => ~[]
        }
      }
    };

    let pandoc = do vec::find(possible_pandocs) |pandoc| {
        let output = process_output(*pandoc, [~"--version"]);
        debug!("testing pandoc cmd %s: %?", *pandoc, output);
        output.status == 0
    };

    if pandoc.is_some() {
        result::Ok(pandoc)
    } else {
        result::Err(~"couldn't find pandoc")
    }
}

#[cfg(test)]
mod test {
    use core::prelude::*;

    use config::*;
    use core::result;
    use core::run::ProcessOutput;

    fn parse_config(args: &[~str]) -> Result<Config, ~str> {
        parse_config_(args, mock_process_output)
    }

    #[test]
    fn should_find_pandoc() {
        let config = Config {
            output_format: PandocHtml,
            .. default_config(&Path("test"))
        };
        let mock_process_output: ~fn(&str, &[~str]) -> ProcessOutput = |_, _| {
            ProcessOutput { status: 0, output: "pandoc 1.8.2.1".to_bytes(), error: ~[] }
        };
        let result = maybe_find_pandoc(&config, None, mock_process_output);
        assert!(result == result::Ok(Some(~"pandoc")));
    }

    #[test]
    fn should_error_with_no_pandoc() {
        let config = Config {
            output_format: PandocHtml,
            .. default_config(&Path("test"))
        };
        let mock_process_output: ~fn(&str, &[~str]) -> ProcessOutput = |_, _| {
            ProcessOutput { status: 1, output: ~[], error: ~[] }
        };
        let result = maybe_find_pandoc(&config, None, mock_process_output);
        assert!(result == result::Err(~"couldn't find pandoc"));
    }

    #[test]
    fn should_error_with_no_crates() {
        let config = parse_config([~"rustdoc"]);
        assert!(config.get_err() == ~"no crates specified");
    }

    #[test]
    fn should_error_with_multiple_crates() {
        let config =
            parse_config([~"rustdoc", ~"crate1.rc", ~"crate2.rc"]);
        assert!(config.get_err() == ~"multiple crates specified");
    }

    #[test]
    fn should_set_output_dir_to_cwd_if_not_provided() {
        let config = parse_config([~"rustdoc", ~"crate.rc"]);
        assert!(config.get().output_dir == Path("."));
    }

    #[test]
    fn should_set_output_dir_if_provided() {
        let config = parse_config([
            ~"rustdoc", ~"crate.rc", ~"--output-dir", ~"snuggles"
        ]);
        assert!(config.get().output_dir == Path("snuggles"));
    }

    #[test]
    fn should_set_output_format_to_pandoc_html_if_not_provided() {
        let config = parse_config([~"rustdoc", ~"crate.rc"]);
        assert!(config.get().output_format == PandocHtml);
    }

    #[test]
    fn should_set_output_format_to_markdown_if_requested() {
        let config = parse_config([
            ~"rustdoc", ~"crate.rc", ~"--output-format", ~"markdown"
        ]);
        assert!(config.get().output_format == Markdown);
    }

    #[test]
    fn should_set_output_format_to_pandoc_html_if_requested() {
        let config = parse_config([
            ~"rustdoc", ~"crate.rc", ~"--output-format", ~"html"
        ]);
        assert!(config.get().output_format == PandocHtml);
    }

    #[test]
    fn should_error_on_bogus_format() {
        let config = parse_config([
            ~"rustdoc", ~"crate.rc", ~"--output-format", ~"bogus"
        ]);
        assert!(config.get_err() == ~"unknown output format 'bogus'");
    }

    #[test]
    fn should_set_output_style_to_doc_per_mod_by_default() {
        let config = parse_config([~"rustdoc", ~"crate.rc"]);
        assert!(config.get().output_style == DocPerMod);
    }

    #[test]
    fn should_set_output_style_to_one_doc_if_requested() {
        let config = parse_config([
            ~"rustdoc", ~"crate.rc", ~"--output-style", ~"doc-per-crate"
        ]);
        assert!(config.get().output_style == DocPerCrate);
    }

    #[test]
    fn should_set_output_style_to_doc_per_mod_if_requested() {
        let config = parse_config([
            ~"rustdoc", ~"crate.rc", ~"--output-style", ~"doc-per-mod"
        ]);
        assert!(config.get().output_style == DocPerMod);
    }

    #[test]
    fn should_error_on_bogus_output_style() {
        let config = parse_config([
            ~"rustdoc", ~"crate.rc", ~"--output-style", ~"bogus"
        ]);
        assert!(config.get_err() == ~"unknown output style 'bogus'");
    }

    #[test]
    fn should_set_pandoc_command_if_requested() {
        let config = parse_config([
            ~"rustdoc", ~"crate.rc", ~"--pandoc-cmd", ~"panda-bear-doc"
        ]);
        assert!(config.get().pandoc_cmd == Some(~"panda-bear-doc"));
    }

    #[test]
    fn should_set_pandoc_command_when_using_pandoc() {
        let config = parse_config([~"rustdoc", ~"crate.rc"]);
        assert!(config.get().pandoc_cmd == Some(~"pandoc"));
    }
}
