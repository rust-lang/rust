use result::Result;
use std::getopts;
use std::cell::Cell;

/// The type of document to output
pub enum OutputFormat {
    /// Markdown
    pub Markdown,
    /// HTML, via markdown and pandoc
    pub PandocHtml
}

impl OutputFormat : cmp::Eq {
    pure fn eq(&self, other: &OutputFormat) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &OutputFormat) -> bool { !(*self).eq(other) }
}

/// How to organize the output
pub enum OutputStyle {
    /// All in a single document
    pub DocPerCrate,
    /// Each module in its own document
    pub DocPerMod
}

impl OutputStyle : cmp::Eq {
    pure fn eq(&self, other: &OutputStyle) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &OutputStyle) -> bool { !(*self).eq(other) }
}

/// The configuration for a rustdoc session
pub type Config = {
    input_crate: Path,
    output_dir: Path,
    output_format: OutputFormat,
    output_style: OutputStyle,
    pandoc_cmd: Option<~str>
};

impl Config: Clone {
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
    use io::println;

    println(~"Usage: rustdoc [options] <cratefile>\n");
    println(~"Options:\n");
    for opts().each |opt| {
        println(fmt!("    %s", opt.second()));
    }
    println(~"");
}

pub fn default_config(input_crate: &Path) -> Config {
    {
        input_crate: *input_crate,
        output_dir: Path("."),
        output_format: PandocHtml,
        output_style: DocPerMod,
        pandoc_cmd: None
    }
}

type ProgramOutput = fn~((&str), (&[~str])) ->
    {status: int, out: ~str, err: ~str};

fn mock_program_output(_prog: &str, _args: &[~str]) -> {
    status: int, out: ~str, err: ~str
} {
    {
        status: 0,
        out: ~"",
        err: ~""
    }
}

pub fn parse_config(args: &[~str]) -> Result<Config, ~str> {
    parse_config_(args, run::program_output)
}

fn parse_config_(
    args: &[~str],
    +program_output: ProgramOutput
) -> Result<Config, ~str> {
    let args = args.tail();
    let opts = vec::unzip(opts()).first();
    match getopts::getopts(args, opts) {
        result::Ok(matches) => {
            if matches.free.len() == 1 {
                let input_crate = Path(vec::head(matches.free));
                config_from_opts(&input_crate, &matches, move program_output)
            } else if matches.free.is_empty() {
                result::Err(~"no crates specified")
            } else {
                result::Err(~"multiple crates specified")
            }
        }
        result::Err(f) => {
            result::Err(getopts::fail_str(f))
        }
    }
}

fn config_from_opts(
    input_crate: &Path,
    matches: &getopts::Matches,
    +program_output: ProgramOutput
) -> Result<Config, ~str> {

    let config = default_config(input_crate);
    let result = result::Ok(config);
    let result = do result::chain(result) |config| {
        let output_dir = getopts::opt_maybe_str(matches, opt_output_dir());
        let output_dir = output_dir.map(|s| Path(*s));
        result::Ok({
            output_dir: output_dir.get_default(config.output_dir),
            .. config
        })
    };
    let result = do result::chain(result) |config| {
        let output_format = getopts::opt_maybe_str(
            matches, opt_output_format());
        do output_format.map_default(result::Ok(config))
            |output_format| {
            do result::chain(parse_output_format(*output_format))
                |output_format| {

                result::Ok({
                    output_format: output_format,
                    .. config
                })
            }
        }
    };
    let result = do result::chain(result) |config| {
        let output_style =
            getopts::opt_maybe_str(matches, opt_output_style());
        do output_style.map_default(result::Ok(config))
            |output_style| {
            do result::chain(parse_output_style(*output_style))
                |output_style| {
                result::Ok({
                    output_style: output_style,
                    .. config
                })
            }
        }
    };
    let program_output = Cell(move program_output);
    let result = do result::chain(result) |config| {
        let pandoc_cmd = getopts::opt_maybe_str(matches, opt_pandoc_cmd());
        let pandoc_cmd = maybe_find_pandoc(
            &config, pandoc_cmd, move program_output.take());
        do result::chain(pandoc_cmd) |pandoc_cmd| {
            result::Ok({
                pandoc_cmd: pandoc_cmd,
                .. config
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

fn maybe_find_pandoc(
    config: &Config,
    +maybe_pandoc_cmd: Option<~str>,
    +program_output: ProgramOutput
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
        let output = program_output(*pandoc, ~[~"--version"]);
        debug!("testing pandoc cmd %s: %?", *pandoc, output);
        output.status == 0
    };

    if pandoc.is_some() {
        result::Ok(pandoc)
    } else {
        result::Err(~"couldn't find pandoc")
    }
}

#[test]
fn should_find_pandoc() {
    let config = {
        output_format: PandocHtml,
        .. default_config(&Path("test"))
    };
    let mock_program_output = fn~(_prog: &str, _args: &[~str]) -> {
        status: int, out: ~str, err: ~str
    } {
        {
            status: 0, out: ~"pandoc 1.8.2.1", err: ~""
        }
    };
    let result = maybe_find_pandoc(&config, None, move mock_program_output);
    assert result == result::Ok(Some(~"pandoc"));
}

#[test]
fn should_error_with_no_pandoc() {
    let config = {
        output_format: PandocHtml,
        .. default_config(&Path("test"))
    };
    let mock_program_output = fn~(_prog: &str, _args: &[~str]) -> {
        status: int, out: ~str, err: ~str
    } {
        {
            status: 1, out: ~"", err: ~""
        }
    };
    let result = maybe_find_pandoc(&config, None, move mock_program_output);
    assert result == result::Err(~"couldn't find pandoc");
}

#[cfg(test)]
mod test {
    #[legacy_exports];
    fn parse_config(args: &[~str]) -> Result<Config, ~str> {
        parse_config_(args, mock_program_output)
    }
}

#[test]
fn should_error_with_no_crates() {
    let config = test::parse_config(~[~"rustdoc"]);
    assert config.get_err() == ~"no crates specified";
}

#[test]
fn should_error_with_multiple_crates() {
    let config =
        test::parse_config(~[~"rustdoc", ~"crate1.rc", ~"crate2.rc"]);
    assert config.get_err() == ~"multiple crates specified";
}

#[test]
fn should_set_output_dir_to_cwd_if_not_provided() {
    let config = test::parse_config(~[~"rustdoc", ~"crate.rc"]);
    assert config.get().output_dir == Path(".");
}

#[test]
fn should_set_output_dir_if_provided() {
    let config = test::parse_config(~[
        ~"rustdoc", ~"crate.rc", ~"--output-dir", ~"snuggles"
    ]);
    assert config.get().output_dir == Path("snuggles");
}

#[test]
fn should_set_output_format_to_pandoc_html_if_not_provided() {
    let config = test::parse_config(~[~"rustdoc", ~"crate.rc"]);
    assert config.get().output_format == PandocHtml;
}

#[test]
fn should_set_output_format_to_markdown_if_requested() {
    let config = test::parse_config(~[
        ~"rustdoc", ~"crate.rc", ~"--output-format", ~"markdown"
    ]);
    assert config.get().output_format == Markdown;
}

#[test]
fn should_set_output_format_to_pandoc_html_if_requested() {
    let config = test::parse_config(~[
        ~"rustdoc", ~"crate.rc", ~"--output-format", ~"html"
    ]);
    assert config.get().output_format == PandocHtml;
}

#[test]
fn should_error_on_bogus_format() {
    let config = test::parse_config(~[
        ~"rustdoc", ~"crate.rc", ~"--output-format", ~"bogus"
    ]);
    assert config.get_err() == ~"unknown output format 'bogus'";
}

#[test]
fn should_set_output_style_to_doc_per_mod_by_default() {
    let config = test::parse_config(~[~"rustdoc", ~"crate.rc"]);
    assert config.get().output_style == DocPerMod;
}

#[test]
fn should_set_output_style_to_one_doc_if_requested() {
    let config = test::parse_config(~[
        ~"rustdoc", ~"crate.rc", ~"--output-style", ~"doc-per-crate"
    ]);
    assert config.get().output_style == DocPerCrate;
}

#[test]
fn should_set_output_style_to_doc_per_mod_if_requested() {
    let config = test::parse_config(~[
        ~"rustdoc", ~"crate.rc", ~"--output-style", ~"doc-per-mod"
    ]);
    assert config.get().output_style == DocPerMod;
}

#[test]
fn should_error_on_bogus_output_style() {
    let config = test::parse_config(~[
        ~"rustdoc", ~"crate.rc", ~"--output-style", ~"bogus"
    ]);
    assert config.get_err() == ~"unknown output style 'bogus'";
}

#[test]
fn should_set_pandoc_command_if_requested() {
    let config = test::parse_config(~[
        ~"rustdoc", ~"crate.rc", ~"--pandoc-cmd", ~"panda-bear-doc"
    ]);
    assert config.get().pandoc_cmd == Some(~"panda-bear-doc");
}

#[test]
fn should_set_pandoc_command_when_using_pandoc() {
    let config = test::parse_config(~[~"rustdoc", ~"crate.rc"]);
    assert config.get().pandoc_cmd == Some(~"pandoc");
}
