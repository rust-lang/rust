use getopts::Options;

#[derive(Debug)]
pub struct Config {
    /// The directory documentation output was generated in
    pub doc_dir: String,
    /// The file documentation was generated for, with docck directives to check
    pub template: String,
}

/// Create a Config from a vector of command-line arguments
pub fn parse_config(args: Vec<String>) -> Config {
    let mut opts = Options::new();
    opts.reqopt("", "doc-dir", "Path to the documentation directory", "PATH")
        .reqopt("", "template", "Path to the template file", "PATH")
        .optflag("h", "help", "show this message");

    let (argv0, args_) = args.split_first().unwrap();
    if args.len() == 1 {
        let message = format!("Usage: {} <doc-dir> <template>", argv0);
        println!("{}", opts.usage(&message));
        std::process::exit(1);
    }

    let matches = opts.parse(args_).unwrap();

    if matches.opt_present("h") || matches.opt_present("help") {
        let message = format!("Usage: {} <doc-dir> <template>", argv0);
        println!("{}", opts.usage(&message));
        std::process::exit(1);
    }

    Config {
        doc_dir: matches.opt_str("doc-dir").unwrap(),
        template: matches.opt_str("template").unwrap(),
    }
}
