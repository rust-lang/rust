use std::cell::LazyCell;
use std::env::Args;

use getopts::Options;

#[derive(Debug)]
pub struct Config {
    /// The directory documentation output was generated in.
    pub doc_dir: String,
    /// The file documentation was generated for, with `jsondocck` directives to check.
    pub template: String,
}

/// Create [`Config`] from an iterator of command-line arguments.
pub fn parse_config(mut args: Args) -> Option<Config> {
    let mut opts = Options::new();
    opts.reqopt("", "doc-dir", "Path to the documentation output directory.", "PATH")
        .reqopt("", "template", "Path to the input template file.", "PATH")
        .optflag("h", "help", "Show this message.");

    let argv0 = args.next().unwrap();
    let usage = &*LazyCell::new(|| opts.usage(&format!("Usage: {argv0} <doc-dir> <template>")));

    if args.len() == 0 {
        print!("{usage}");

        return None;
    }

    let matches = opts.parse(args).unwrap();

    if matches.opt_present("h") || matches.opt_present("help") {
        print!("{usage}");

        return None;
    }

    Some(Config {
        doc_dir: matches.opt_str("doc-dir").unwrap(),
        template: matches.opt_str("template").unwrap(),
    })
}
