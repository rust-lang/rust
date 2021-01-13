use clap::crate_version;

use std::env;
use std::path::{Path, PathBuf};

use clap::{App, AppSettings, ArgMatches, SubCommand};

use mdbook::errors::Result as Result3;
use mdbook::MDBook;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let d_message = "-d, --dest-dir=[dest-dir]
'The output directory for your book{n}(Defaults to ./book when omitted)'";
    let dir_message = "[dir]
'A directory for your book{n}(Defaults to Current Directory when omitted)'";

    let matches = App::new("rustbook")
        .about("Build a book with mdBook")
        .author("Steve Klabnik <steve@steveklabnik.com>")
        .version(&*format!("v{}", crate_version!()))
        .setting(AppSettings::SubcommandRequired)
        .subcommand(
            SubCommand::with_name("build")
                .about("Build the book from the markdown files")
                .arg_from_usage(d_message)
                .arg_from_usage(dir_message),
        )
        .subcommand(
            SubCommand::with_name("test")
                .about("Tests that a book's Rust code samples compile")
                .arg_from_usage(dir_message),
        )
        .get_matches();

    // Check which subcomamnd the user ran...
    match matches.subcommand() {
        ("build", Some(sub_matches)) => {
            if let Err(e) = build(sub_matches) {
                handle_error(e);
            }
        }
        ("test", Some(sub_matches)) => {
            if let Err(e) = test(sub_matches) {
                handle_error(e);
            }
        }
        (_, _) => unreachable!(),
    };
}

// Build command implementation
pub fn build(args: &ArgMatches<'_>) -> Result3<()> {
    let book_dir = get_book_dir(args);
    let mut book = load_book(&book_dir)?;

    // Set this to allow us to catch bugs in advance.
    book.config.build.create_missing = false;

    if let Some(dest_dir) = args.value_of("dest-dir") {
        book.config.build.build_dir = PathBuf::from(dest_dir);
    }

    book.build()?;

    Ok(())
}

fn test(args: &ArgMatches<'_>) -> Result3<()> {
    let book_dir = get_book_dir(args);
    let mut book = load_book(&book_dir)?;
    book.test(vec![])
}

fn get_book_dir(args: &ArgMatches<'_>) -> PathBuf {
    if let Some(dir) = args.value_of("dir") {
        // Check if path is relative from current dir, or absolute...
        let p = Path::new(dir);
        if p.is_relative() { env::current_dir().unwrap().join(dir) } else { p.to_path_buf() }
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
