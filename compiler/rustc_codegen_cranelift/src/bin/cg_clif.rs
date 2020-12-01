#![feature(rustc_private)]

extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_session;
extern crate rustc_target;

use rustc_data_structures::profiling::print_time_passes_entry;
use rustc_interface::interface;
use rustc_session::config::ErrorOutputType;
use rustc_session::early_error;
use rustc_target::spec::PanicStrategy;

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
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_owned()
        }));
    }
}

fn main() {
    let start = std::time::Instant::now();
    rustc_driver::init_rustc_env_logger();
    let mut callbacks = CraneliftPassesCallbacks::default();
    rustc_driver::install_ice_hook();
    let exit_code = rustc_driver::catch_with_exit_code(|| {
        let mut use_jit = false;

        let mut args = std::env::args_os()
            .enumerate()
            .map(|(i, arg)| {
                arg.into_string().unwrap_or_else(|arg| {
                    early_error(
                        ErrorOutputType::default(),
                        &format!("Argument {} is not valid Unicode: {:?}", i, arg),
                    )
                })
            })
            .filter(|arg| {
                if arg == "--jit" {
                    use_jit = true;
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();
        if use_jit {
            args.push("-Cprefer-dynamic".to_string());
        }
        let mut run_compiler = rustc_driver::RunCompiler::new(&args, &mut callbacks);
        run_compiler.set_make_codegen_backend(Some(Box::new(move |_| {
            Box::new(rustc_codegen_cranelift::CraneliftCodegenBackend {
                config: rustc_codegen_cranelift::BackendConfig { use_jit },
            })
        })));
        run_compiler.run()
    });
    // The extra `\t` is necessary to align this label with the others.
    print_time_passes_entry(callbacks.time_passes, "\ttotal", start.elapsed());
    std::process::exit(exit_code)
}
