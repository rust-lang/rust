use std::path::PathBuf;

use crate::error::DiagCtxt;

pub(crate) struct Config {
    /// The path to the directory that contains the generated HTML documentation.
    pub(crate) doc_dir: PathBuf,
    /// The path to the test file the docs were generated for and which may contain check commands.
    pub(crate) template: String,
    /// Whether to automatically update snapshot files.
    #[allow(dead_code)] // FIXME
    pub(crate) bless: bool,
}

impl Config {
    pub(crate) fn parse(args: &[String], dcx: &mut DiagCtxt) -> Result<Self, ()> {
        const DOC_DIR_OPT: &str = "doc-dir";
        const TEMPLATE_OPT: &str = "template";
        const BLESS_FLAG: &str = "bless";

        let mut opts = getopts::Options::new();
        opts.reqopt("", DOC_DIR_OPT, "Path to the documentation directory", "<PATH>")
            .reqopt("", TEMPLATE_OPT, "Path to the template file", "<PATH>")
            .optflag("", BLESS_FLAG, "Whether to automatically update snapshot files");

        // We may not assume the presence of the first argument. On some platforms,
        // it's possible to pass an empty array of arguments to `execve`.
        let program = args.get(0).map(|arg| arg.as_str()).unwrap_or("htmldocck");
        let args = args.get(1..).unwrap_or_default();

        match opts.parse(args) {
            Ok(matches) => Ok(Self {
                doc_dir: matches.opt_str(DOC_DIR_OPT).unwrap().into(),
                template: matches.opt_str(TEMPLATE_OPT).unwrap(),
                bless: matches.opt_present(BLESS_FLAG),
            }),
            Err(err) => {
                let mut err = err.to_string();
                err.push_str("\n\n");
                err.push_str(&opts.short_usage(program));
                err.push_str(&opts.usage(""));
                dcx.emit(&err, None, None);
                Err(())
            }
        }
    }
}
