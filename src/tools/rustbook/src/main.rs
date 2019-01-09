//
extern crate mdbook;
#[macro_use]
extern crate clap;

use std::env;
use std::path::{Path, PathBuf};

use clap::{App, ArgMatches, SubCommand, AppSettings};

use mdbook::MDBook;
use mdbook::errors::Result;

fn main() {
    let d_message = "-d, --dest-dir=[dest-dir]
'The output directory for your book{n}(Defaults to ./book when omitted)'";
    let dir_message = "[dir]
'A directory for your book{n}(Defaults to Current Directory when omitted)'";

    let matches = App::new("rustbook")
                    .about("Build a book with mdBook")
                    .author("Steve Klabnik <steve@steveklabnik.com>")
                    .version(&*format!("v{}", crate_version!()))
                    .setting(AppSettings::SubcommandRequired)
                    .subcommand(SubCommand::with_name("build")
                        .about("Build the book from the markdown files")
                        .arg_from_usage(d_message)
                        .arg_from_usage(dir_message))
                    .get_matches();

    // Check which subcomamnd the user ran...
    let res = match matches.subcommand() {
        ("build", Some(sub_matches)) => build(sub_matches),
        (_, _) => unreachable!(),
    };

    if let Err(e) = res {
        eprintln!("Error: {}", e);

        for cause in e.iter().skip(1) {
            eprintln!("\tCaused By: {}", cause);
        }

        ::std::process::exit(101);
    }
}
// Build command implementation
pub fn build(args: &ArgMatches) -> Result<()> {
    let book_dir = get_book_dir(args);
    let mut book = MDBook::load(&book_dir)?;

    // Set this to allow us to catch bugs in advance.
    book.config.build.create_missing = false;

    if let Some(dest_dir) = args.value_of("dest-dir") {
        book.config.build.build_dir = PathBuf::from(dest_dir);
    }

    book.build()?;

    Ok(())
}

fn get_book_dir(args: &ArgMatches) -> PathBuf {
    if let Some(dir) = args.value_of("dir") {
        // Check if path is relative from current dir, or absolute...
        let p = Path::new(dir);
        if p.is_relative() {
            env::current_dir().unwrap().join(dir)
        } else {
            p.to_path_buf()
        }
    } else {
        env::current_dir().unwrap()
    }
}
