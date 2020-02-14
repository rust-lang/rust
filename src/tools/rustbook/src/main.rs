use clap::crate_version;

use std::env;
use std::path::{Path, PathBuf};

use clap::{App, AppSettings, ArgMatches, SubCommand};

use mdbook::errors::Result as Result3;
use mdbook::MDBook;

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
        .subcommand(
            SubCommand::with_name("build")
                .about("Build the book from the markdown files")
                .arg_from_usage(d_message)
                .arg_from_usage(dir_message),
        )
        .subcommand(
            SubCommand::with_name("linkcheck")
                .about("Run linkcheck with mdBook 3")
                .arg_from_usage(dir_message),
        )
        .get_matches();

    // Check which subcomamnd the user ran...
    match matches.subcommand() {
        ("build", Some(sub_matches)) => {
            if let Err(e) = build(sub_matches) {
                eprintln!("Error: {}", e);

                for cause in e.iter().skip(1) {
                    eprintln!("\tCaused By: {}", cause);
                }

                ::std::process::exit(101);
            }
        }
        ("linkcheck", Some(sub_matches)) => {
            #[cfg(feature = "linkcheck")]
            {
                let (diags, files) = linkcheck(sub_matches).expect("Error while linkchecking.");
                if !diags.is_empty() {
                    let color = codespan_reporting::term::termcolor::ColorChoice::Auto;
                    let mut writer =
                        codespan_reporting::term::termcolor::StandardStream::stderr(color);
                    let cfg = codespan_reporting::term::Config::default();

                    for diag in diags {
                        codespan_reporting::term::emit(&mut writer, &cfg, &files, &diag)
                            .expect("Unable to emit linkcheck error.");
                    }

                    std::process::exit(101);
                }
            }

            #[cfg(not(feature = "linkcheck"))]
            {
                // This avoids the `unused_binding` lint.
                println!(
                    "mdbook-linkcheck is disabled, but arguments were passed: {:?}",
                    sub_matches
                );
            }
        }
        (_, _) => unreachable!(),
    };
}

#[cfg(feature = "linkcheck")]
pub fn linkcheck(
    args: &ArgMatches<'_>,
) -> Result<(Vec<codespan_reporting::diagnostic::Diagnostic>, codespan::Files), failure::Error> {
    use mdbook_linkcheck::Reason;

    let book_dir = get_book_dir(args);
    let src_dir = book_dir.join("src");
    let book = MDBook::load(&book_dir).unwrap();
    let linkck_cfg = mdbook_linkcheck::get_config(&book.config)?;
    let mut files = codespan::Files::new();
    let target_files = mdbook_linkcheck::load_files_into_memory(&book.book, &mut files);
    let cache = mdbook_linkcheck::Cache::default();

    let (links, incomplete) = mdbook_linkcheck::extract_links(target_files, &files);

    let outcome =
        mdbook_linkcheck::validate(&links, &linkck_cfg, &src_dir, &cache, &files, incomplete)?;

    let mut is_real_error = false;

    for link in outcome.invalid_links.iter() {
        match &link.reason {
            Reason::FileNotFound | Reason::TraversesParentDirectories => {
                is_real_error = true;
            }
            Reason::UnsuccessfulServerResponse(status) => {
                if status.is_client_error() {
                    is_real_error = true;
                } else {
                    eprintln!("Unsuccessful server response for link `{}`", link.link.uri);
                }
            }
            Reason::Client(err) => {
                if err.is_timeout() {
                    eprintln!("Timeout for link `{}`", link.link.uri);
                } else if err.is_server_error() {
                    eprintln!("Server error for link `{}`", link.link.uri);
                } else if !err.is_http() {
                    eprintln!("Non-HTTP-related error for link: {} {}", link.link.uri, err);
                } else {
                    is_real_error = true;
                }
            }
        }
    }

    if is_real_error {
        Ok((outcome.generate_diagnostics(&files, linkck_cfg.warning_policy), files))
    } else {
        Ok((vec![], files))
    }
}

// Build command implementation
pub fn build(args: &ArgMatches<'_>) -> Result3<()> {
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
        if p.is_relative() { env::current_dir().unwrap().join(dir) } else { p.to_path_buf() }
    } else {
        env::current_dir().unwrap()
    }
}
