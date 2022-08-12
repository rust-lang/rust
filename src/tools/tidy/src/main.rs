//! Tidy checks source code in this repository.
//!
//! This program runs all of the various tidy checks for style, cleanliness,
//! etc. This is run by default on `./x.py test` and as part of the auto
//! builders. The tidy checks can be executed with `./x.py test tidy`.

use tidy::*;

use crossbeam_utils::thread::{scope, ScopedJoinHandle};
use std::collections::VecDeque;
use std::env;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};

fn main() {
    let root_path: PathBuf = env::args_os().nth(1).expect("need path to root of repo").into();
    let cargo: PathBuf = env::args_os().nth(2).expect("need path to cargo").into();
    let output_directory: PathBuf =
        env::args_os().nth(3).expect("need path to output directory").into();
    let concurrency: NonZeroUsize =
        FromStr::from_str(&env::args().nth(4).expect("need concurrency"))
            .expect("concurrency must be a number");

    let paths = Paths::from_root(&root_path);

    let args: Vec<String> = env::args().skip(1).collect();

    let verbose = args.iter().any(|s| *s == "--verbose");

    let bad = std::sync::Arc::new(AtomicBool::new(false));

    scope(|s| {
        let mut handles: VecDeque<ScopedJoinHandle<'_, ()>> =
            VecDeque::with_capacity(concurrency.get());

        macro_rules! check {
            ($p:ident $(, $args:expr)* ) => {
                while handles.len() >= concurrency.get() {
                    handles.pop_front().unwrap().join().unwrap();
                }

                let handle = s.spawn(|_| {
                    let mut flag = false;
                    $p::check($($args),* , &mut flag);
                    if (flag) {
                        bad.store(true, Ordering::Relaxed);
                    }
                });
                handles.push_back(handle);
            }
        }

        check!(target_specific_tests, &paths.src);

        // Checks that are done on the cargo workspace.
        check!(deps, &root_path, &cargo);
        check!(extdeps, &root_path);

        // Checks over tests.
        check!(debug_artifacts, &paths.src);
        check!(ui_tests, &paths.src);

        // Checks that only make sense for the compiler.
        check!(errors, &paths.compiler);
        check!(error_codes_check, &[&paths.src, &paths.compiler]);

        // Checks that only make sense for the std libs.
        check!(pal, &paths.library);
        check!(primitive_docs, &paths.library);

        // Checks that need to be done for both the compiler and std libraries.
        check!(unit_tests, &paths.src);
        check!(unit_tests, &paths.compiler);
        check!(unit_tests, &paths.library);

        if bins::check_filesystem_support(&[&root_path], &output_directory) {
            check!(bins, &root_path);
        }

        check!(style, &paths.src);
        check!(style, &paths.compiler);
        check!(style, &paths.library);

        check!(edition, &paths.src);
        check!(edition, &paths.compiler);
        check!(edition, &paths.library);

        let collected = {
            while handles.len() >= concurrency.get() {
                handles.pop_front().unwrap().join().unwrap();
            }
            let mut flag = false;
            let r = features::check(&paths, &mut flag, verbose);
            if flag {
                bad.store(true, Ordering::Relaxed);
            }
            r
        };
        check!(unstable_book, &paths.src, collected);
    })
    .unwrap();

    if bad.load(Ordering::Relaxed) {
        eprintln!("some tidy checks failed");
        process::exit(1);
    }
}
