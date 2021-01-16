use getopts::Options;

#[derive(Debug)]
pub struct Config {
    /// The directory documentation output was generated in
    pub doc_dir: String,
    /// The file documentation was generated for, with docck commands to check
    pub template: String,
}

pub fn parse_config(args: Vec<String>) -> Config {
    let mut opts = Options::new();
    opts.reqopt("", "doc-dir", "Path to the documentation directory", "PATH")
        .reqopt("", "template", "Path to the template file", "PATH")
        .optflag("h", "help", "show this message");

    let (argv0, args_) = args.split_first().unwrap();
    if args.len() == 1 || args[1] == "-h" || args[1] == "--help" {
        let message = format!("Usage: {} <doc-dir> <template>", argv0);
        println!("{}", opts.usage(&message));
        println!();
        panic!()
    }

    let matches = &match opts.parse(args_) {
        Ok(m) => m,
        Err(f) => panic!("{:?}", f),
    };

    if matches.opt_present("h") || matches.opt_present("help") {
        let message = format!("Usage: {} <doc-dir> <template>", argv0);
        println!("{}", opts.usage(&message));
        println!();
        panic!()
    }

    Config {
        doc_dir: matches.opt_str("doc-dir").unwrap(),
        template: matches.opt_str("template").unwrap(),
    }
}
