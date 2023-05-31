//! Tidy checks source code in this repository.
//!
//! This program runs all of the various tidy checks for style, cleanliness,
//! etc. This is run by default on `./x.py test` and as part of the auto
//! builders. The tidy checks can be executed with `./x.py test tidy`.

use tidy::*;

use std::collections::VecDeque;
use std::env;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, scope, ScopedJoinHandle};

fn main() {
    // Running Cargo will read the libstd Cargo.toml
    // which uses the unstable `public-dependency` feature.
    //
    // `setenv` might not be thread safe, so run it before using multiple threads.
    env::set_var("RUSTC_BOOTSTRAP", "1");

    let root_path: PathBuf = env::args_os().nth(1).expect("need path to root of repo").into();
    let cargo: PathBuf = env::args_os().nth(2).expect("need path to cargo").into();
    let output_directory: PathBuf =
        env::args_os().nth(3).expect("need path to output directory").into();
    let concurrency: NonZeroUsize =
        FromStr::from_str(&env::args().nth(4).expect("need concurrency"))
            .expect("concurrency must be a number");

    let src_path = root_path.join("src");
    let tests_path = root_path.join("tests");
    let library_path = root_path.join("library");
    let compiler_path = root_path.join("compiler");
    let librustdoc_path = src_path.join("librustdoc");

    let args: Vec<String> = env::args().skip(1).collect();

    let verbose = args.iter().any(|s| *s == "--verbose");
    let bless = args.iter().any(|s| *s == "--bless");

    let bad = std::sync::Arc::new(AtomicBool::new(false));

    let drain_handles = |handles: &mut VecDeque<ScopedJoinHandle<'_, ()>>| {
        // poll all threads for completion before awaiting the oldest one
        for i in (0..handles.len()).rev() {
            if handles[i].is_finished() {
                handles.swap_remove_back(i).unwrap().join().unwrap();
            }
        }

        while handles.len() >= concurrency.get() {
            handles.pop_front().unwrap().join().unwrap();
        }
    };

    scope(|s| {
        let mut handles: VecDeque<ScopedJoinHandle<'_, ()>> =
            VecDeque::with_capacity(concurrency.get());

        macro_rules! check {
            ($p:ident) => {
                check!(@ $p, name=format!("{}", stringify!($p)));
            };
            ($p:ident, $path:expr $(, $args:expr)* ) => {
                let shortened = $path.strip_prefix(&root_path).unwrap();
                let name = if shortened == std::path::Path::new("") {
                    format!("{} (.)", stringify!($p))
                } else {
                    format!("{} ({})", stringify!($p), shortened.display())
                };
                check!(@ $p, name=name, $path $(,$args)*);
            };
            (@ $p:ident, name=$name:expr $(, $args:expr)* ) => {
                drain_handles(&mut handles);

                let handle = thread::Builder::new().name($name).spawn_scoped(s, || {
                    let mut flag = false;
                    $p::check($($args, )* &mut flag);
                    if (flag) {
                        bad.store(true, Ordering::Relaxed);
                    }
                }).unwrap();
                handles.push_back(handle);
            }
        }

        check!(target_specific_tests, &tests_path);

        // Checks that are done on the cargo workspace.
        check!(deps, &root_path, &cargo);
        check!(extdeps, &root_path);

        // Checks over tests.
        check!(tests_placement, &root_path);
        check!(debug_artifacts, &tests_path);
        check!(ui_tests, &tests_path);
        check!(mir_opt_tests, &tests_path, bless);
        check!(rustdoc_gui_tests, &tests_path);

        // Checks that only make sense for the compiler.
        check!(error_codes, &root_path, &[&compiler_path, &librustdoc_path], verbose);
        check!(fluent_alphabetical, &compiler_path, bless);

        // Checks that only make sense for the std libs.
        check!(pal, &library_path);
        check!(primitive_docs, &library_path);

        // Checks that need to be done for both the compiler and std libraries.
        check!(unit_tests, &src_path);
        check!(unit_tests, &compiler_path);
        check!(unit_tests, &library_path);

        if bins::check_filesystem_support(&[&root_path], &output_directory) {
            check!(bins, &root_path);
        }

        check!(style, &src_path);
        check!(style, &tests_path);
        check!(style, &compiler_path);
        check!(style, &library_path);

        check!(edition, &src_path);
        check!(edition, &compiler_path);
        check!(edition, &library_path);

        check!(alphabetical, &src_path);
        check!(alphabetical, &compiler_path);
        check!(alphabetical, &library_path);

        check!(x_version, &root_path, &cargo);

        let collected = {
            drain_handles(&mut handles);

            let mut flag = false;
            let r = features::check(
                &src_path,
                &tests_path,
                &compiler_path,
                &library_path,
                &mut flag,
                verbose,
            );
            if flag {
                bad.store(true, Ordering::Relaxed);
            }
            r
        };
        check!(unstable_book, &src_path, collected);
    });

    if bad.load(Ordering::Relaxed) {
        eprintln!("some tidy checks failed");
        process::exit(1);
    }
}
