#![feature(rustc_private)]
#![warn(unused_extern_crates)]

use std::process::ExitCode;
use std::sync::OnceLock;

use rustc_lint::LintStore;
use rustc_session::Session;

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_lint;
extern crate rustc_session;

struct Callbacks;

type LintRegisterFunc = unsafe extern "C" fn(&Session, &mut LintStore);

static LINT_LIBRARIES: OnceLock<Vec<libloading::Library>> = OnceLock::new();

impl rustc_driver::Callbacks for Callbacks {
    fn config(&mut self, config: &mut rustc_interface::Config) {
        let previous = config.register_lints.take();
        config.register_lints = Some(Box::new(move |sess, lint_store| {
            if let Some(previous) = &previous {
                previous(sess, lint_store);
            }
            for library in LINT_LIBRARIES.get().unwrap() {
                unsafe {
                    let register = library.get::<LintRegisterFunc>(b"register_lints").unwrap();
                    (*register)(sess, lint_store);
                }
            }
        }))
    }
}

fn main() -> ExitCode {
    rustc_driver::catch_with_exit_code(|| {
        let libraries = std::env::var("COMPILETEST_LINT_DRIVER_PATHS")
            .expect("`COMPILETEST_LINT_DRIVER_PATHS` not provided")
            .split(':')
            .map(|path| unsafe { libloading::Library::new(path).unwrap() })
            .collect();
        LINT_LIBRARIES.set(libraries).unwrap();
        let args = std::env::args().collect::<Vec<String>>();
        rustc_driver::run_compiler(&args, &mut Callbacks);
    })
}
