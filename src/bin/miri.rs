#![feature(rustc_private, custom_attribute)]
#![allow(unused_attributes)]

extern crate getopts;
extern crate miri;
extern crate rustc;
extern crate rustc_driver;
extern crate env_logger;
extern crate log_settings;
extern crate syntax;
#[macro_use] extern crate log;

use miri::{
    EvalContext,
    CachedMir,
    step,
    EvalError,
    Frame,
};
use rustc::session::Session;
use rustc_driver::{driver, CompilerCalls};
use rustc::ty::{TyCtxt, subst};
use rustc::mir::mir_map::MirMap;
use rustc::hir::def_id::DefId;

struct MiriCompilerCalls;

impl<'a> CompilerCalls<'a> for MiriCompilerCalls {
    fn build_controller(
        &mut self,
        _: &Session,
        _: &getopts::Matches
    ) -> driver::CompileController<'a> {
        let mut control = driver::CompileController::basic();

        control.after_analysis.callback = Box::new(|state| {
            state.session.abort_if_errors();
            interpret_start_points(state.tcx.unwrap(), state.mir_map.unwrap());
        });

        control
    }
}



fn interpret_start_points<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir_map: &MirMap<'tcx>,
) {
    let initial_indentation = ::log_settings::settings().indentation;
    for (&id, mir) in &mir_map.map {
        for attr in tcx.map.attrs(id) {
            use syntax::attr::AttrMetaMethods;
            if attr.check_name("miri_run") {
                let item = tcx.map.expect_item(id);

                ::log_settings::settings().indentation = initial_indentation;

                debug!("Interpreting: {}", item.name);

                let mut gecx = EvalContext::new(tcx, mir_map);
                let substs = tcx.mk_substs(subst::Substs::empty());
                let return_ptr = gecx.alloc_ret_ptr(mir.return_ty, substs);

                gecx.push_stack_frame(tcx.map.local_def_id(id), mir.span, CachedMir::Ref(mir), substs, return_ptr);

                loop { match (step(&mut gecx), return_ptr) {
                    (Ok(true), _) => {},
                    (Ok(false), Some(ptr)) => if log_enabled!(::log::LogLevel::Debug) {
                        gecx.memory().dump(ptr.alloc_id);
                        break;
                    },
                    (Ok(false), None) => {
                        warn!("diverging function returned");
                        break;
                    },
                    // FIXME: diverging functions can end up here in some future miri
                    (Err(e), _) => {
                        report(tcx, &gecx, e);
                        break;
                    },
                } }
            }
        }
    }
}

fn report(tcx: TyCtxt, gecx: &EvalContext, e: EvalError) {
    let frame = gecx.stack().last().expect("stackframe was empty");
    let block = frame.mir.basic_block_data(frame.next_block);
    let span = if frame.stmt < block.statements.len() {
        block.statements[frame.stmt].span
    } else {
        block.terminator().span
    };
    let mut err = tcx.sess.struct_span_err(span, &e.to_string());
    for &Frame { def_id, substs, span, .. } in gecx.stack().iter().rev() {
        // FIXME(solson): Find a way to do this without this Display impl hack.
        use rustc::util::ppaux;
        use std::fmt;
        struct Instance<'tcx>(DefId, &'tcx subst::Substs<'tcx>);
        impl<'tcx> fmt::Display for Instance<'tcx> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                ppaux::parameterized(f, self.1, self.0, ppaux::Ns::Value, &[],
                    |tcx| tcx.lookup_item_type(self.0).generics)
            }
        }
        err.span_note(span, &format!("inside call to {}", Instance(def_id, substs)));
    }
    err.emit();
}

#[miri_run]
fn main() {
    init_logger();
    let args: Vec<String> = std::env::args().collect();
    rustc_driver::run_compiler(&args, &mut MiriCompilerCalls);
}

#[miri_run]
fn init_logger() {
    const NSPACES: usize = 40;
    let format = |record: &log::LogRecord| {
        // prepend spaces to indent the final string
        let indentation = log_settings::settings().indentation;
        format!("{lvl}:{module}{depth:2}{indent:<indentation$} {text}",
            lvl = record.level(),
            module = record.location().module_path(),
            depth = indentation / NSPACES,
            indentation = indentation % NSPACES,
            indent = "",
            text = record.args())
    };

    let mut builder = env_logger::LogBuilder::new();
    builder.format(format).filter(None, log::LogLevelFilter::Info);

    if std::env::var("MIRI_LOG").is_ok() {
        builder.parse(&std::env::var("MIRI_LOG").unwrap());
    }

    builder.init().unwrap();
}
