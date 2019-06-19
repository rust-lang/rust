#![deny(rust_2018_idioms)]

use clap::{crate_version};

use std::env;
use std::path::{Path, PathBuf};

use clap::{App, ArgMatches, SubCommand, AppSettings};

use mdbook_1::{MDBook as MDBook1};
use mdbook_1::errors::{Result as Result1};

use mdbook::MDBook;
use mdbook::errors::Result;

fn main() {
    let d_message = "-d, --dest-dir=[dest-dir]
'The output directory for your book{n}(Defaults to ./book when omitted)'";
    let dir_message = "[dir]
'A directory for your book{n}(Defaults to Current Directory when omitted)'";
    let vers_message = "-m, --mdbook-vers=[md-version]
'The version of mdbook to use for your book{n}(Defaults to 1 when omitted)'";

    let matches = App::new("rustbook")
                    .about("Build a book with mdBook")
                    .author("Steve Klabnik <steve@steveklabnik.com>")
                    .version(&*format!("v{}", crate_version!()))
                    .setting(AppSettings::SubcommandRequired)
                    .subcommand(SubCommand::with_name("build")
                        .about("Build the book from the markdown files")
                        .arg_from_usage(d_message)
                        .arg_from_usage(dir_message)
                        .arg_from_usage(vers_message))
                    .get_matches();

    // Check which subcomamnd the user ran...
    match matches.subcommand() {
        ("build", Some(sub_matches)) => {
            match sub_matches.value_of("mdbook-vers") {
                None | Some("1") => {
                    if let Err(e) = build_1(sub_matches) {
                        eprintln!("Error: {}", e);

                        for cause in e.iter().skip(1) {
                            eprintln!("\tCaused By: {}", cause);
                        }

                        ::std::process::exit(101);
                    }
                }
                Some("2") | Some("3") => {
                    if let Err(e) = build(sub_matches) {
                        eprintln!("Error: {}", e);

                        for cause in e.iter().skip(1) {
                            eprintln!("\tCaused By: {}", cause);
                        }

                        ::std::process::exit(101);
                    }
                }
                _ => {
                    panic!("Invalid mdBook version! Select '1' or '2' or '3'");
                }
            };
        },
        (_, _) => unreachable!(),
    };
}

// Build command implementation
pub fn build_1(args: &ArgMatches<'_>) -> Result1<()> {
    let book_dir = get_book_dir(args);
    let mut book = MDBook1::load(&book_dir)?;

    // Set this to allow us to catch bugs in advance.
    book.config.build.create_missing = false;

    if let Some(dest_dir) = args.value_of("dest-dir") {
        book.config.build.build_dir = PathBuf::from(dest_dir);
    }

    book.build()?;

    Ok(())
}

// Build command implementation
pub fn build(args: &ArgMatches<'_>) -> Result<()> {
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

fn get_book_dir(args: &ArgMatches<'_>) -> PathBuf {
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
