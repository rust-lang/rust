#![feature(rustc_private)]
extern crate miri;
extern crate getopts;
extern crate rustc;
extern crate rustc_metadata;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_codegen_utils;
extern crate rustc_interface;
extern crate syntax;

use std::path::Path;
use std::io::Write;
use std::sync::{Mutex, Arc};
use std::io;


use rustc_interface::interface;
use rustc::hir::{self, itemlikevisit};
use rustc::ty::TyCtxt;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc_driver::Compilation;

use miri::MiriConfig;

struct MiriCompilerCalls {
    /// whether we are building for the host
    host_target: bool,
}

impl rustc_driver::Callbacks for MiriCompilerCalls {
    fn after_parsing(&mut self, compiler: &interface::Compiler) -> Compilation {
        let attr = (
            syntax::symbol::Symbol::intern("miri"),
            syntax::feature_gate::AttributeType::Whitelisted,
        );
        compiler.session().plugin_attributes.borrow_mut().push(attr);

        Compilation::Continue
    }

    fn after_analysis(&mut self, compiler: &interface::Compiler) -> Compilation {
        compiler.session().abort_if_errors();
        compiler.global_ctxt().unwrap().peek_mut().enter(|tcx| {
            if std::env::args().any(|arg| arg == "--test") {
                struct Visitor<'tcx>(TyCtxt<'tcx>);
                impl<'tcx, 'hir> itemlikevisit::ItemLikeVisitor<'hir> for Visitor<'tcx> {
                    fn visit_item(&mut self, i: &'hir hir::Item) {
                        if let hir::ItemKind::Fn(.., body_id) = i.kind {
                            if i.attrs.iter().any(|attr| attr.check_name(syntax::symbol::sym::test)) {
                                let config = MiriConfig {
                                    validate: true,
                                    communicate: false,
                                    excluded_env_vars: vec![],
                                    args: vec![],
                                    seed: None,
                                };
                                let did = self.0.hir().body_owner_def_id(body_id);
                                println!("running test: {}", self.0.def_path_debug_str(did));
                                miri::eval_main(self.0, did, config);
                                self.0.sess.abort_if_errors();
                            }
                        }
                    }
                    fn visit_trait_item(&mut self, _trait_item: &'hir hir::TraitItem) {}
                    fn visit_impl_item(&mut self, _impl_item: &'hir hir::ImplItem) {}
                }
                tcx.hir().krate().visit_all_item_likes(&mut Visitor(tcx));
            } else if let Some((entry_def_id, _)) = tcx.entry_fn(LOCAL_CRATE) {
                let config = MiriConfig {
                    validate: true,
                    communicate: false,
                    excluded_env_vars: vec![],
                    args: vec![],
                    seed: None
                };
                miri::eval_main(tcx, entry_def_id, config);

                compiler.session().abort_if_errors();
            } else {
                println!("no main function found, assuming auxiliary build");
            }
        });

        // Continue execution on host target
        if self.host_target { Compilation::Continue } else { Compilation::Stop }
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
        args.splice(1..1, miri::miri_default_args().iter().map(ToString::to_string));
        // file to process
        args.push(path.display().to_string());

        let sysroot_flag = String::from("--sysroot");
        if !args.contains(&sysroot_flag) {
            args.push(sysroot_flag);
            args.push(Path::new(&std::env::var("HOME").unwrap()).join(".xargo").join("HOST").display().to_string());
        }

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
            let _ = rustc_driver::run_compiler(&args, &mut MiriCompilerCalls { host_target }, None, Some(Box::new(buf)));
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
