#![feature(rustc_private, once_cell)]
#![warn(rust_2018_idioms)]
#![warn(unused_lifetimes)]
#![warn(unreachable_pub)]

extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_session;
extern crate rustc_target;

use std::lazy::SyncLazy;
use std::panic;

use rustc_data_structures::profiling::{get_resident_set_size, print_time_passes_entry};
use rustc_interface::interface;
use rustc_session::config::ErrorOutputType;
use rustc_session::early_error;
use rustc_target::spec::PanicStrategy;

const BUG_REPORT_URL: &str = "https://github.com/bjorn3/rustc_codegen_cranelift/issues/new";

static DEFAULT_HOOK: SyncLazy<Box<dyn Fn(&panic::PanicInfo<'_>) + Sync + Send + 'static>> =
    SyncLazy::new(|| {
        let hook = panic::take_hook();
        panic::set_hook(Box::new(|info| {
            // Invoke the default handler, which prints the actual panic message and optionally a backtrace
            (*DEFAULT_HOOK)(info);

            // Separate the output with an empty line
            eprintln!();

            // Print the ICE message
            rustc_driver::report_ice(info, BUG_REPORT_URL);
        }));
        hook
    });

#[derive(Default)]
pub struct CraneliftPassesCallbacks {
    time_passes: bool,
}

impl rustc_driver::Callbacks for CraneliftPassesCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        // If a --prints=... option has been given, we don't print the "total"
        // time because it will mess up the --prints output. See #64339.
        self.time_passes = config.opts.prints.is_empty()
            && (config.opts.debugging_opts.time_passes || config.opts.debugging_opts.time);

        config.opts.cg.panic = Some(PanicStrategy::Abort);
        config.opts.debugging_opts.panic_abort_tests = true;
        config.opts.maybe_sysroot = Some(config.opts.maybe_sysroot.clone().unwrap_or_else(|| {
            std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().to_owned()
        }));
    }
}

fn main() {
    let start_time = std::time::Instant::now();
    let start_rss = get_resident_set_size();
    rustc_driver::init_rustc_env_logger();
    let mut callbacks = CraneliftPassesCallbacks::default();
    SyncLazy::force(&DEFAULT_HOOK); // Install ice hook
    let exit_code = rustc_driver::catch_with_exit_code(|| {
        let args = std::env::args_os()
            .enumerate()
            .map(|(i, arg)| {
                arg.into_string().unwrap_or_else(|arg| {
                    early_error(
                        ErrorOutputType::default(),
                        &format!("Argument {} is not valid Unicode: {:?}", i, arg),
                    )
                })
            })
            .collect::<Vec<_>>();
        let mut run_compiler = rustc_driver::RunCompiler::new(&args, &mut callbacks);
        run_compiler.set_make_codegen_backend(Some(Box::new(move |_| {
            Box::new(rustc_codegen_cranelift::CraneliftCodegenBackend { config: None })
        })));
        run_compiler.run()
    });

    if callbacks.time_passes {
        let end_rss = get_resident_set_size();
        print_time_passes_entry("total", start_time.elapsed(), start_rss, end_rss);
    }

    std::process::exit(exit_code)
}
