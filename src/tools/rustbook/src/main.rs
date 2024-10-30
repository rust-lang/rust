use std::env;
use std::path::{Path, PathBuf};

use clap::{ArgMatches, Command, arg, crate_version};
use mdbook::MDBook;
use mdbook::errors::Result as Result3;
use mdbook_i18n_helpers::preprocessors::Gettext;
use mdbook_spec::Spec;
use mdbook_trpl_listing::TrplListing;
use mdbook_trpl_note::TrplNote;

fn main() {
    let crate_version = concat!("v", crate_version!());
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let d_arg = arg!(-d --"dest-dir" <DEST_DIR>
"The output directory for your book\n(Defaults to ./book when omitted)")
    .required(false)
    .value_parser(clap::value_parser!(PathBuf));

    let l_arg = arg!(-l --"lang" <LANGUAGE>
"The output language")
    .required(false)
    .value_parser(clap::value_parser!(String));

    let root_arg = arg!(--"rust-root" <ROOT_DIR>
"Path to the root of the rust source tree")
    .required(false)
    .value_parser(clap::value_parser!(PathBuf));

    let dir_arg = arg!([dir] "Root directory for the book\n\
                              (Defaults to the current directory when omitted)")
    .value_parser(clap::value_parser!(PathBuf));

    let matches = Command::new("rustbook")
        .about("Build a book with mdBook")
        .author("Steve Klabnik <steve@steveklabnik.com>")
        .version(crate_version)
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("build")
                .about("Build the book from the markdown files")
                .arg(d_arg)
                .arg(l_arg)
                .arg(root_arg)
                .arg(&dir_arg),
        )
        .subcommand(
            Command::new("test")
                .about("Tests that a book's Rust code samples compile")
                .arg(dir_arg),
        )
        .get_matches();

    // Check which subcomamnd the user ran...
    match matches.subcommand() {
        Some(("build", sub_matches)) => {
            if let Err(e) = build(sub_matches) {
                handle_error(e);
            }
        }
        Some(("test", sub_matches)) => {
            if let Err(e) = test(sub_matches) {
                handle_error(e);
            }
        }
        _ => unreachable!(),
    };
}

// Build command implementation
pub fn build(args: &ArgMatches) -> Result3<()> {
    let book_dir = get_book_dir(args);
    let mut book = load_book(&book_dir)?;

    if let Some(lang) = args.get_one::<String>("lang") {
        let gettext = Gettext;
        book.with_preprocessor(gettext);
        book.config.set("book.language", lang).unwrap();
    }

    // Set this to allow us to catch bugs in advance.
    book.config.build.create_missing = false;

    if let Some(dest_dir) = args.get_one::<PathBuf>("dest-dir") {
        book.config.build.build_dir = dest_dir.into();
    }

    // NOTE: Replacing preprocessors using this technique causes error
    // messages to be displayed when the original preprocessor doesn't work
    // (but it otherwise succeeds).
    //
    // This should probably be fixed in mdbook to remove the existing
    // preprocessor, or this should modify the config and use
    // MDBook::load_with_config.
    if book.config.get_preprocessor("trpl-note").is_some() {
        book.with_preprocessor(TrplNote);
    }

    if book.config.get_preprocessor("trpl-listing").is_some() {
        book.with_preprocessor(TrplListing);
    }

    if book.config.get_preprocessor("spec").is_some() {
        let rust_root = args.get_one::<PathBuf>("rust-root").cloned();
        book.with_preprocessor(Spec::new(rust_root)?);
    }

    book.build()?;

    Ok(())
}

fn test(args: &ArgMatches) -> Result3<()> {
    let book_dir = get_book_dir(args);
    let mut book = load_book(&book_dir)?;
    book.test(vec![])
}

fn get_book_dir(args: &ArgMatches) -> PathBuf {
    if let Some(p) = args.get_one::<PathBuf>("dir") {
        // Check if path is relative from current dir, or absolute...
        if p.is_relative() { env::current_dir().unwrap().join(p) } else { p.to_path_buf() }
    } else {
        env::current_dir().unwrap()
    }
}

fn load_book(book_dir: &Path) -> Result3<MDBook> {
    let mut book = MDBook::load(book_dir)?;
    book.config.set("output.html.input-404", "").unwrap();
    Ok(book)
}

fn handle_error(error: mdbook::errors::Error) -> ! {
    eprintln!("Error: {}", error);

    for cause in error.chain().skip(1) {
        eprintln!("\tCaused By: {}", cause);
    }

    ::std::process::exit(101);
}
