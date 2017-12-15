#![feature(rustc_private, i128_type)]
extern crate miri;
extern crate getopts;
extern crate rustc;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate syntax;

use std::path::{PathBuf, Path};
use std::io::Write;
use std::sync::{Mutex, Arc};
use std::io;


use rustc::session::Session;
use rustc::middle::cstore::CrateStore;
use rustc_driver::{Compilation, CompilerCalls, RustcDefaultCalls};
use rustc_driver::driver::{CompileState, CompileController};
use rustc::session::config::{self, Input, ErrorOutputType};
use rustc::hir::{self, itemlikevisit};
use rustc::ty::TyCtxt;
use syntax::ast;

struct MiriCompilerCalls {
    default: RustcDefaultCalls,
    /// whether we are building for the host
    host_target: bool,
}

impl<'a> CompilerCalls<'a> for MiriCompilerCalls {
    fn early_callback(
        &mut self,
        matches: &getopts::Matches,
        sopts: &config::Options,
        cfg: &ast::CrateConfig,
        descriptions: &rustc_errors::registry::Registry,
        output: ErrorOutputType
    ) -> Compilation {
        self.default.early_callback(matches, sopts, cfg, descriptions, output)
    }
    fn no_input(
        &mut self,
        matches: &getopts::Matches,
        sopts: &config::Options,
        cfg: &ast::CrateConfig,
        odir: &Option<PathBuf>,
        ofile: &Option<PathBuf>,
        descriptions: &rustc_errors::registry::Registry
    ) -> Option<(Input, Option<PathBuf>)> {
        self.default.no_input(matches, sopts, cfg, odir, ofile, descriptions)
    }
    fn late_callback(
        &mut self,
        matches: &getopts::Matches,
        sess: &Session,
        cstore: &CrateStore,
        input: &Input,
        odir: &Option<PathBuf>,
        ofile: &Option<PathBuf>
    ) -> Compilation {
        self.default.late_callback(matches, sess, cstore, input, odir, ofile)
    }
    fn build_controller(&mut self, sess: &Session, matches: &getopts::Matches) -> CompileController<'a> {
        let mut control = self.default.build_controller(sess, matches);
        control.after_hir_lowering.callback = Box::new(after_hir_lowering);
        control.after_analysis.callback = Box::new(after_analysis);
        if !self.host_target {
            // only fully compile targets on the host
            control.after_analysis.stop = Compilation::Stop;
        }
        control
    }
}

fn after_hir_lowering(state: &mut CompileState) {
    let attr = (String::from("miri"), syntax::feature_gate::AttributeType::Whitelisted);
    state.session.plugin_attributes.borrow_mut().push(attr);
}

fn after_analysis<'a, 'tcx>(state: &mut CompileState<'a, 'tcx>) {
    state.session.abort_if_errors();

    let tcx = state.tcx.unwrap();
    let limits = Default::default();

    if std::env::args().any(|arg| arg == "--test") {
        struct Visitor<'a, 'tcx: 'a>(miri::ResourceLimits, TyCtxt<'a, 'tcx, 'tcx>, &'a CompileState<'a, 'tcx>);
        impl<'a, 'tcx: 'a, 'hir> itemlikevisit::ItemLikeVisitor<'hir> for Visitor<'a, 'tcx> {
            fn visit_item(&mut self, i: &'hir hir::Item) {
                if let hir::Item_::ItemFn(_, _, _, _, _, body_id) = i.node {
                    if i.attrs.iter().any(|attr| attr.name().map_or(false, |n| n == "test")) {
                        let did = self.1.hir.body_owner_def_id(body_id);
                        println!("running test: {}", self.1.def_path_debug_str(did));
                        miri::eval_main(self.1, did, None, self.0);
                        self.2.session.abort_if_errors();
                    }
                }
            }
            fn visit_trait_item(&mut self, _trait_item: &'hir hir::TraitItem) {}
            fn visit_impl_item(&mut self, _impl_item: &'hir hir::ImplItem) {}
        }
        state.hir_crate.unwrap().visit_all_item_likes(&mut Visitor(limits, tcx, state));
    } else if let Some((entry_node_id, _)) = *state.session.entry_fn.borrow() {
        let entry_def_id = tcx.hir.local_def_id(entry_node_id);
        let start_wrapper = tcx.lang_items().start_fn().and_then(|start_fn|
                                if tcx.is_mir_available(start_fn) { Some(start_fn) } else { None });
        miri::eval_main(tcx, entry_def_id, start_wrapper, limits);

        state.session.abort_if_errors();
    } else {
        println!("no main function found, assuming auxiliary build");
    }
}

fn main() {
    let path = option_env!("MIRI_RUSTC_TEST")
        .map(String::from)
        .unwrap_or_else(|| {
            std::env::var("MIRI_RUSTC_TEST")
                .expect("need to set MIRI_RUSTC_TEST to path of rustc tests")
        });

    let mut mir_not_found = Vec::new();
    let mut crate_not_found = Vec::new();
    let mut success = 0;
    let mut failed = Vec::new();
    let mut c_abi_fns = Vec::new();
    let mut abi = Vec::new();
    let mut unsupported = Vec::new();
    let mut unimplemented_intrinsic = Vec::new();
    let mut limits = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(path).unwrap().collect();
    while let Some(file) = files.pop() {
        let file = file.unwrap();
        let path = file.path();
        if file.metadata().unwrap().is_dir() {
            if !path.to_str().unwrap().ends_with("auxiliary") {
                // add subdirs recursively
                files.extend(std::fs::read_dir(path).unwrap());
            }
            continue;
        }
        if !file.metadata().unwrap().is_file() || !path.to_str().unwrap().ends_with(".rs") {
            continue;
        }
        let stderr = std::io::stderr();
        write!(stderr.lock(), "test [miri-pass] {} ... ", path.display()).unwrap();
        let mut host_target = false;
        let mut args: Vec<String> = std::env::args().filter(|arg| {
            if arg == "--miri_host_target" {
                host_target = true;
                false // remove the flag, rustc doesn't know it
            } else {
                true
            }
        }).collect();
        // file to process
        args.push(path.display().to_string());

        let sysroot_flag = String::from("--sysroot");
        if !args.contains(&sysroot_flag) {
            args.push(sysroot_flag);
            args.push(Path::new(&std::env::var("HOME").unwrap()).join(".xargo").join("HOST").display().to_string());
        }

        args.push("-Zmir-opt-level=3".to_owned());
        // for auxilary builds in unit tests
        args.push("-Zalways-encode-mir".to_owned());

        // A threadsafe buffer for writing.
        #[derive(Default, Clone)]
        struct BufWriter(Arc<Mutex<Vec<u8>>>);

        impl Write for BufWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().write(buf)
            }
            fn flush(&mut self) -> io::Result<()> {
                self.0.lock().unwrap().flush()
            }
        }
        let buf = BufWriter::default();
        let output = buf.clone();
        let result = std::panic::catch_unwind(|| {
            rustc_driver::run_compiler(&args, &mut MiriCompilerCalls {
                default: RustcDefaultCalls,
                host_target,
            }, None, Some(Box::new(buf)));
        });

        match result {
            Ok(()) => {
                success += 1;
                writeln!(stderr.lock(), "ok").unwrap()
            },
            Err(_) => {
                let output = output.0.lock().unwrap();
                let output_err = std::str::from_utf8(&output).unwrap();
                if let Some(text) = output_err.splitn(2, "no mir for `").nth(1) {
                    let end = text.find('`').unwrap();
                    mir_not_found.push(text[..end].to_string());
                    writeln!(stderr.lock(), "NO MIR FOR `{}`", &text[..end]).unwrap();
                } else if let Some(text) = output_err.splitn(2, "can't find crate for `").nth(1) {
                    let end = text.find('`').unwrap();
                    crate_not_found.push(text[..end].to_string());
                    writeln!(stderr.lock(), "CAN'T FIND CRATE FOR `{}`", &text[..end]).unwrap();
                } else {
                    for text in output_err.split("error: ").skip(1) {
                        let end = text.find('\n').unwrap_or(text.len());
                        let c_abi = "can't call C ABI function: ";
                        let unimplemented_intrinsic_s = "unimplemented intrinsic: ";
                        let unsupported_s = "miri does not support ";
                        let abi_s = "can't handle function with ";
                        let limit_s = "reached the configured maximum ";
                        if text.starts_with(c_abi) {
                            c_abi_fns.push(text[c_abi.len()..end].to_string());
                        } else if text.starts_with(unimplemented_intrinsic_s) {
                            unimplemented_intrinsic.push(text[unimplemented_intrinsic_s.len()..end].to_string());
                        } else if text.starts_with(unsupported_s) {
                            unsupported.push(text[unsupported_s.len()..end].to_string());
                        } else if text.starts_with(abi_s) {
                            abi.push(text[abi_s.len()..end].to_string());
                        } else if text.starts_with(limit_s) {
                            limits.push(text[limit_s.len()..end].to_string());
                        } else if text.find("aborting").is_none() {
                            failed.push(text[..end].to_string());
                        }
                    }
                    writeln!(stderr.lock(), "stderr: \n {}", output_err).unwrap();
                }
            }
        }
    }
    let stderr = std::io::stderr();
    let mut stderr = stderr.lock();
    writeln!(stderr, "{} success, {} no mir, {} crate not found, {} failed, \
                        {} C fn, {} ABI, {} unsupported, {} intrinsic",
                        success, mir_not_found.len(), crate_not_found.len(), failed.len(),
                        c_abi_fns.len(), abi.len(), unsupported.len(), unimplemented_intrinsic.len()).unwrap();
    writeln!(stderr, "# The \"other reasons\" errors").unwrap();
    writeln!(stderr, "(sorted, deduplicated)").unwrap();
    print_vec(&mut stderr, failed);

    writeln!(stderr, "# can't call C ABI function").unwrap();
    print_vec(&mut stderr, c_abi_fns);

    writeln!(stderr, "# unsupported ABI").unwrap();
    print_vec(&mut stderr, abi);

    writeln!(stderr, "# unsupported").unwrap();
    print_vec(&mut stderr, unsupported);

    writeln!(stderr, "# unimplemented intrinsics").unwrap();
    print_vec(&mut stderr, unimplemented_intrinsic);

    writeln!(stderr, "# mir not found").unwrap();
    print_vec(&mut stderr, mir_not_found);

    writeln!(stderr, "# crate not found").unwrap();
    print_vec(&mut stderr, crate_not_found);
}

fn print_vec<W: std::io::Write>(stderr: &mut W, v: Vec<String>) {
    writeln!(stderr, "```").unwrap();
    for (n, s) in vec_to_hist(v).into_iter().rev() {
        writeln!(stderr, "{:4} {}", n, s).unwrap();
    }
    writeln!(stderr, "```").unwrap();
}

fn vec_to_hist<T: PartialEq + Ord>(mut v: Vec<T>) -> Vec<(usize, T)> {
    v.sort();
    let mut v = v.into_iter();
    let mut result = Vec::new();
    let mut current = v.next();
    'outer: while let Some(current_val) = current {
        let mut n = 1;
        for next in &mut v {
            if next == current_val {
                n += 1;
            } else {
                result.push((n, current_val));
                current = Some(next);
                continue 'outer;
            }
        }
        result.push((n, current_val));
        break;
    }
    result.sort();
    result
}
