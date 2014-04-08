// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::archive::{Archive, METADATA_FILENAME};
use back::rpath;
use back::svh::Svh;
use driver::driver::{CrateTranslation, OutputFilenames};
use driver::session::{NoDebugInfo, Session};
use driver::session;
use lib::llvm::llvm;
use lib::llvm::ModuleRef;
use lib;
use metadata::common::LinkMeta;
use metadata::{encoder, cstore, filesearch, csearch};
use middle::trans::context::CrateContext;
use middle::trans::common::gensym_name;
use middle::ty;
use util::common::time;
use util::ppaux;
use util::sha2::{Digest, Sha256};

use std::c_str::{ToCStr, CString};
use std::char;
use std::os::consts::{macos, freebsd, linux, android, win32};
use std::ptr;
use std::str;
use std::io;
use std::io::{fs, TempDir, Process};
use flate;
use serialize::hex::ToHex;
use syntax::abi;
use syntax::ast;
use syntax::ast_map::{PathElem, PathElems, PathName};
use syntax::ast_map;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::crateid::CrateId;
use syntax::parse::token;

#[deriving(Clone, Eq, Ord, TotalOrd, TotalEq)]
pub enum OutputType {
    OutputTypeBitcode,
    OutputTypeAssembly,
    OutputTypeLlvmAssembly,
    OutputTypeObject,
    OutputTypeExe,
}

pub fn llvm_err(sess: &Session, msg: ~str) -> ! {
    unsafe {
        let cstr = llvm::LLVMRustGetLastError();
        if cstr == ptr::null() {
            sess.fatal(msg);
        } else {
            let err = CString::new(cstr, false);
            let err = str::from_utf8_lossy(err.as_bytes());
            sess.fatal(msg + ": " + err.as_slice());
        }
    }
}

pub fn WriteOutputFile(
        sess: &Session,
        target: lib::llvm::TargetMachineRef,
        pm: lib::llvm::PassManagerRef,
        m: ModuleRef,
        output: &Path,
        file_type: lib::llvm::FileType) {
    unsafe {
        output.with_c_str(|output| {
            let result = llvm::LLVMRustWriteOutputFile(
                    target, pm, m, output, file_type);
            if !result {
                llvm_err(sess, ~"could not write output");
            }
        })
    }
}

pub mod write {

    use back::lto;
    use back::link::{WriteOutputFile, OutputType};
    use back::link::{OutputTypeAssembly, OutputTypeBitcode};
    use back::link::{OutputTypeExe, OutputTypeLlvmAssembly};
    use back::link::{OutputTypeObject};
    use driver::driver::{CrateTranslation, OutputFilenames};
    use driver::session::{NoDebugInfo, Session};
    use driver::session;
    use lib::llvm::llvm;
    use lib::llvm::{ModuleRef, TargetMachineRef, PassManagerRef};
    use lib;
    use util::common::time;
    use syntax::abi;

    use std::c_str::ToCStr;
    use std::io::Process;
    use libc::{c_uint, c_int};
    use std::str;

    // On android, we by default compile for armv7 processors. This enables
    // things like double word CAS instructions (rather than emulating them)
    // which are *far* more efficient. This is obviously undesirable in some
    // cases, so if any sort of target feature is specified we don't append v7
    // to the feature list.
    fn target_feature<'a>(sess: &'a Session) -> &'a str {
        match sess.targ_cfg.os {
            abi::OsAndroid => {
                if "" == sess.opts.cg.target_feature {
                    "+v7"
                } else {
                    sess.opts.cg.target_feature.as_slice()
                }
            }
            _ => sess.opts.cg.target_feature.as_slice()
        }
    }

    pub fn run_passes(sess: &Session,
                      trans: &CrateTranslation,
                      output_types: &[OutputType],
                      output: &OutputFilenames) {
        let llmod = trans.module;
        let llcx = trans.context;
        unsafe {
            configure_llvm(sess);

            if sess.opts.cg.save_temps {
                output.with_extension("no-opt.bc").with_c_str(|buf| {
                    llvm::LLVMWriteBitcodeToFile(llmod, buf);
                })
            }

            let opt_level = match sess.opts.optimize {
              session::No => lib::llvm::CodeGenLevelNone,
              session::Less => lib::llvm::CodeGenLevelLess,
              session::Default => lib::llvm::CodeGenLevelDefault,
              session::Aggressive => lib::llvm::CodeGenLevelAggressive,
            };
            let use_softfp = sess.opts.cg.soft_float;

            // FIXME: #11906: Omitting frame pointers breaks retrieving the value of a parameter.
            // FIXME: #11954: mac64 unwinding may not work with fp elim
            let no_fp_elim = (sess.opts.debuginfo != NoDebugInfo) ||
                             (sess.targ_cfg.os == abi::OsMacos &&
                              sess.targ_cfg.arch == abi::X86_64);

            let reloc_model = match sess.opts.cg.relocation_model.as_slice() {
                "pic" => lib::llvm::RelocPIC,
                "static" => lib::llvm::RelocStatic,
                "default" => lib::llvm::RelocDefault,
                "dynamic-no-pic" => lib::llvm::RelocDynamicNoPic,
                _ => {
                    sess.err(format!("{} is not a valid relocation mode",
                             sess.opts.cg.relocation_model));
                    sess.abort_if_errors();
                    return;
                }
            };

            let tm = sess.targ_cfg.target_strs.target_triple.with_c_str(|t| {
                sess.opts.cg.target_cpu.with_c_str(|cpu| {
                    target_feature(sess).with_c_str(|features| {
                        llvm::LLVMRustCreateTargetMachine(
                            t, cpu, features,
                            lib::llvm::CodeModelDefault,
                            reloc_model,
                            opt_level,
                            true,
                            use_softfp,
                            no_fp_elim
                        )
                    })
                })
            });

            // Create the two optimizing pass managers. These mirror what clang
            // does, and are by populated by LLVM's default PassManagerBuilder.
            // Each manager has a different set of passes, but they also share
            // some common passes.
            let fpm = llvm::LLVMCreateFunctionPassManagerForModule(llmod);
            let mpm = llvm::LLVMCreatePassManager();

            // If we're verifying or linting, add them to the function pass
            // manager.
            let addpass = |pass: &str| {
                pass.with_c_str(|s| llvm::LLVMRustAddPass(fpm, s))
            };
            if !sess.no_verify() { assert!(addpass("verify")); }

            if !sess.opts.cg.no_prepopulate_passes {
                llvm::LLVMRustAddAnalysisPasses(tm, fpm, llmod);
                llvm::LLVMRustAddAnalysisPasses(tm, mpm, llmod);
                populate_llvm_passes(fpm, mpm, llmod, opt_level);
            }

            for pass in sess.opts.cg.passes.iter() {
                pass.with_c_str(|s| {
                    if !llvm::LLVMRustAddPass(mpm, s) {
                        sess.warn(format!("unknown pass {}, ignoring", *pass));
                    }
                })
            }

            // Finally, run the actual optimization passes
            time(sess.time_passes(), "llvm function passes", (), |()|
                 llvm::LLVMRustRunFunctionPassManager(fpm, llmod));
            time(sess.time_passes(), "llvm module passes", (), |()|
                 llvm::LLVMRunPassManager(mpm, llmod));

            // Deallocate managers that we're now done with
            llvm::LLVMDisposePassManager(fpm);
            llvm::LLVMDisposePassManager(mpm);

            // Emit the bytecode if we're either saving our temporaries or
            // emitting an rlib. Whenever an rlib is created, the bytecode is
            // inserted into the archive in order to allow LTO against it.
            if sess.opts.cg.save_temps ||
               (sess.crate_types.borrow().contains(&session::CrateTypeRlib) &&
                sess.opts.output_types.contains(&OutputTypeExe)) {
                output.temp_path(OutputTypeBitcode).with_c_str(|buf| {
                    llvm::LLVMWriteBitcodeToFile(llmod, buf);
                })
            }

            if sess.lto() {
                time(sess.time_passes(), "all lto passes", (), |()|
                     lto::run(sess, llmod, tm, trans.reachable.as_slice()));

                if sess.opts.cg.save_temps {
                    output.with_extension("lto.bc").with_c_str(|buf| {
                        llvm::LLVMWriteBitcodeToFile(llmod, buf);
                    })
                }
            }

            // A codegen-specific pass manager is used to generate object
            // files for an LLVM module.
            //
            // Apparently each of these pass managers is a one-shot kind of
            // thing, so we create a new one for each type of output. The
            // pass manager passed to the closure should be ensured to not
            // escape the closure itself, and the manager should only be
            // used once.
            fn with_codegen(tm: TargetMachineRef, llmod: ModuleRef,
                            f: |PassManagerRef|) {
                unsafe {
                    let cpm = llvm::LLVMCreatePassManager();
                    llvm::LLVMRustAddAnalysisPasses(tm, cpm, llmod);
                    llvm::LLVMRustAddLibraryInfo(cpm, llmod);
                    f(cpm);
                    llvm::LLVMDisposePassManager(cpm);
                }
            }

            let mut object_file = None;
            let mut needs_metadata = false;
            for output_type in output_types.iter() {
                let path = output.path(*output_type);
                match *output_type {
                    OutputTypeBitcode => {
                        path.with_c_str(|buf| {
                            llvm::LLVMWriteBitcodeToFile(llmod, buf);
                        })
                    }
                    OutputTypeLlvmAssembly => {
                        path.with_c_str(|output| {
                            with_codegen(tm, llmod, |cpm| {
                                llvm::LLVMRustPrintModule(cpm, llmod, output);
                            })
                        })
                    }
                    OutputTypeAssembly => {
                        // If we're not using the LLVM assembler, this function
                        // could be invoked specially with output_type_assembly,
                        // so in this case we still want the metadata object
                        // file.
                        let ty = OutputTypeAssembly;
                        let path = if sess.opts.output_types.contains(&ty) {
                           path
                        } else {
                            needs_metadata = true;
                            output.temp_path(OutputTypeAssembly)
                        };
                        with_codegen(tm, llmod, |cpm| {
                            WriteOutputFile(sess, tm, cpm, llmod, &path,
                                            lib::llvm::AssemblyFile);
                        });
                    }
                    OutputTypeObject => {
                        object_file = Some(path);
                    }
                    OutputTypeExe => {
                        object_file = Some(output.temp_path(OutputTypeObject));
                        needs_metadata = true;
                    }
                }
            }

            time(sess.time_passes(), "codegen passes", (), |()| {
                match object_file {
                    Some(ref path) => {
                        with_codegen(tm, llmod, |cpm| {
                            WriteOutputFile(sess, tm, cpm, llmod, path,
                                            lib::llvm::ObjectFile);
                        });
                    }
                    None => {}
                }
                if needs_metadata {
                    with_codegen(tm, trans.metadata_module, |cpm| {
                        let out = output.temp_path(OutputTypeObject)
                                        .with_extension("metadata.o");
                        WriteOutputFile(sess, tm, cpm,
                                        trans.metadata_module, &out,
                                        lib::llvm::ObjectFile);
                    })
                }
            });

            llvm::LLVMRustDisposeTargetMachine(tm);
            llvm::LLVMDisposeModule(trans.metadata_module);
            llvm::LLVMDisposeModule(llmod);
            llvm::LLVMContextDispose(llcx);
            if sess.time_llvm_passes() { llvm::LLVMRustPrintPassTimings(); }
        }
    }

    pub fn run_assembler(sess: &Session, outputs: &OutputFilenames) {
        let cc = super::get_cc_prog(sess);
        let assembly = outputs.temp_path(OutputTypeAssembly);
        let object = outputs.path(OutputTypeObject);

        // FIXME (#9639): This needs to handle non-utf8 paths
        let args = [
            ~"-c",
            ~"-o", object.as_str().unwrap().to_owned(),
            assembly.as_str().unwrap().to_owned()];

        debug!("{} '{}'", cc, args.connect("' '"));
        match Process::output(cc, args) {
            Ok(prog) => {
                if !prog.status.success() {
                    sess.err(format!("linking with `{}` failed: {}", cc, prog.status));
                    sess.note(format!("{} arguments: '{}'", cc, args.connect("' '")));
                    let mut note = prog.error.clone();
                    note.push_all(prog.output.as_slice());
                    sess.note(str::from_utf8(note.as_slice()).unwrap().to_owned());
                    sess.abort_if_errors();
                }
            },
            Err(e) => {
                sess.err(format!("could not exec the linker `{}`: {}", cc, e));
                sess.abort_if_errors();
            }
        }
    }

    unsafe fn configure_llvm(sess: &Session) {
        use sync::one::{Once, ONCE_INIT};
        static mut INIT: Once = ONCE_INIT;

        // Copy what clang does by turning on loop vectorization at O2 and
        // slp vectorization at O3
        let vectorize_loop = !sess.opts.cg.no_vectorize_loops &&
                             (sess.opts.optimize == session::Default ||
                              sess.opts.optimize == session::Aggressive);
        let vectorize_slp = !sess.opts.cg.no_vectorize_slp &&
                            sess.opts.optimize == session::Aggressive;

        let mut llvm_c_strs = Vec::new();
        let mut llvm_args = Vec::new();
        {
            let add = |arg: &str| {
                let s = arg.to_c_str();
                llvm_args.push(s.with_ref(|p| p));
                llvm_c_strs.push(s);
            };
            add("rustc"); // fake program name
            if vectorize_loop { add("-vectorize-loops"); }
            if vectorize_slp  { add("-vectorize-slp");   }
            if sess.time_llvm_passes() { add("-time-passes"); }
            if sess.print_llvm_passes() { add("-debug-pass=Structure"); }

            for arg in sess.opts.cg.llvm_args.iter() {
                add(*arg);
            }
        }

        INIT.doit(|| {
            llvm::LLVMInitializePasses();

            // Only initialize the platforms supported by Rust here, because
            // using --llvm-root will have multiple platforms that rustllvm
            // doesn't actually link to and it's pointless to put target info
            // into the registry that Rust cannot generate machine code for.
            llvm::LLVMInitializeX86TargetInfo();
            llvm::LLVMInitializeX86Target();
            llvm::LLVMInitializeX86TargetMC();
            llvm::LLVMInitializeX86AsmPrinter();
            llvm::LLVMInitializeX86AsmParser();

            llvm::LLVMInitializeARMTargetInfo();
            llvm::LLVMInitializeARMTarget();
            llvm::LLVMInitializeARMTargetMC();
            llvm::LLVMInitializeARMAsmPrinter();
            llvm::LLVMInitializeARMAsmParser();

            llvm::LLVMInitializeMipsTargetInfo();
            llvm::LLVMInitializeMipsTarget();
            llvm::LLVMInitializeMipsTargetMC();
            llvm::LLVMInitializeMipsAsmPrinter();
            llvm::LLVMInitializeMipsAsmParser();

            llvm::LLVMRustSetLLVMOptions(llvm_args.len() as c_int,
                                         llvm_args.as_ptr());
        });
    }

    unsafe fn populate_llvm_passes(fpm: lib::llvm::PassManagerRef,
                                   mpm: lib::llvm::PassManagerRef,
                                   llmod: ModuleRef,
                                   opt: lib::llvm::CodeGenOptLevel) {
        // Create the PassManagerBuilder for LLVM. We configure it with
        // reasonable defaults and prepare it to actually populate the pass
        // manager.
        let builder = llvm::LLVMPassManagerBuilderCreate();
        match opt {
            lib::llvm::CodeGenLevelNone => {
                // Don't add lifetime intrinsics at O0
                llvm::LLVMRustAddAlwaysInlinePass(builder, false);
            }
            lib::llvm::CodeGenLevelLess => {
                llvm::LLVMRustAddAlwaysInlinePass(builder, true);
            }
            // numeric values copied from clang
            lib::llvm::CodeGenLevelDefault => {
                llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder,
                                                                    225);
            }
            lib::llvm::CodeGenLevelAggressive => {
                llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder,
                                                                    275);
            }
        }
        llvm::LLVMPassManagerBuilderSetOptLevel(builder, opt as c_uint);
        llvm::LLVMRustAddBuilderLibraryInfo(builder, llmod);

        // Use the builder to populate the function/module pass managers.
        llvm::LLVMPassManagerBuilderPopulateFunctionPassManager(builder, fpm);
        llvm::LLVMPassManagerBuilderPopulateModulePassManager(builder, mpm);
        llvm::LLVMPassManagerBuilderDispose(builder);
    }
}


/*
 * Name mangling and its relationship to metadata. This is complex. Read
 * carefully.
 *
 * The semantic model of Rust linkage is, broadly, that "there's no global
 * namespace" between crates. Our aim is to preserve the illusion of this
 * model despite the fact that it's not *quite* possible to implement on
 * modern linkers. We initially didn't use system linkers at all, but have
 * been convinced of their utility.
 *
 * There are a few issues to handle:
 *
 *  - Linkers operate on a flat namespace, so we have to flatten names.
 *    We do this using the C++ namespace-mangling technique. Foo::bar
 *    symbols and such.
 *
 *  - Symbols with the same name but different types need to get different
 *    linkage-names. We do this by hashing a string-encoding of the type into
 *    a fixed-size (currently 16-byte hex) cryptographic hash function (CHF:
 *    we use SHA256) to "prevent collisions". This is not airtight but 16 hex
 *    digits on uniform probability means you're going to need 2**32 same-name
 *    symbols in the same process before you're even hitting birthday-paradox
 *    collision probability.
 *
 *  - Symbols in different crates but with same names "within" the crate need
 *    to get different linkage-names.
 *
 *  - The hash shown in the filename needs to be predictable and stable for
 *    build tooling integration. It also needs to be using a hash function
 *    which is easy to use from Python, make, etc.
 *
 * So here is what we do:
 *
 *  - Consider the package id; every crate has one (specified with crate_id
 *    attribute).  If a package id isn't provided explicitly, we infer a
 *    versionless one from the output name. The version will end up being 0.0
 *    in this case. CNAME and CVERS are taken from this package id. For
 *    example, github.com/mozilla/CNAME#CVERS.
 *
 *  - Define CMH as SHA256(crateid).
 *
 *  - Define CMH8 as the first 8 characters of CMH.
 *
 *  - Compile our crate to lib CNAME-CMH8-CVERS.so
 *
 *  - Define STH(sym) as SHA256(CMH, type_str(sym))
 *
 *  - Suffix a mangled sym with ::STH@CVERS, so that it is unique in the
 *    name, non-name metadata, and type sense, and versioned in the way
 *    system linkers understand.
 */

pub fn find_crate_id(attrs: &[ast::Attribute], out_filestem: &str) -> CrateId {
    match attr::find_crateid(attrs) {
        None => from_str(out_filestem).unwrap(),
        Some(s) => s,
    }
}

pub fn crate_id_hash(crate_id: &CrateId) -> ~str {
    // This calculates CMH as defined above. Note that we don't use the path of
    // the crate id in the hash because lookups are only done by (name/vers),
    // not by path.
    let mut s = Sha256::new();
    s.input_str(crate_id.short_name_with_version());
    truncated_hash_result(&mut s).slice_to(8).to_owned()
}

pub fn build_link_meta(krate: &ast::Crate, out_filestem: &str) -> LinkMeta {
    let r = LinkMeta {
        crateid: find_crate_id(krate.attrs.as_slice(), out_filestem),
        crate_hash: Svh::calculate(krate),
    };
    info!("{}", r);
    return r;
}

fn truncated_hash_result(symbol_hasher: &mut Sha256) -> ~str {
    let output = symbol_hasher.result_bytes();
    // 64 bits should be enough to avoid collisions.
    output.slice_to(8).to_hex()
}


// This calculates STH for a symbol, as defined above
fn symbol_hash(tcx: &ty::ctxt, symbol_hasher: &mut Sha256,
               t: ty::t, link_meta: &LinkMeta) -> ~str {
    // NB: do *not* use abbrevs here as we want the symbol names
    // to be independent of one another in the crate.

    symbol_hasher.reset();
    symbol_hasher.input_str(link_meta.crateid.name);
    symbol_hasher.input_str("-");
    symbol_hasher.input_str(link_meta.crate_hash.as_str());
    symbol_hasher.input_str("-");
    symbol_hasher.input_str(encoder::encoded_ty(tcx, t));
    let mut hash = truncated_hash_result(symbol_hasher);
    // Prefix with 'h' so that it never blends into adjacent digits
    hash.unshift_char('h');
    hash
}

fn get_symbol_hash(ccx: &CrateContext, t: ty::t) -> ~str {
    match ccx.type_hashcodes.borrow().find(&t) {
        Some(h) => return h.to_str(),
        None => {}
    }

    let mut symbol_hasher = ccx.symbol_hasher.borrow_mut();
    let hash = symbol_hash(ccx.tcx(), &mut *symbol_hasher, t, &ccx.link_meta);
    ccx.type_hashcodes.borrow_mut().insert(t, hash.clone());
    hash
}


// Name sanitation. LLVM will happily accept identifiers with weird names, but
// gas doesn't!
// gas accepts the following characters in symbols: a-z, A-Z, 0-9, ., _, $
pub fn sanitize(s: &str) -> ~str {
    let mut result = ~"";
    for c in s.chars() {
        match c {
            // Escape these with $ sequences
            '@' => result.push_str("$SP$"),
            '~' => result.push_str("$UP$"),
            '*' => result.push_str("$RP$"),
            '&' => result.push_str("$BP$"),
            '<' => result.push_str("$LT$"),
            '>' => result.push_str("$GT$"),
            '(' => result.push_str("$LP$"),
            ')' => result.push_str("$RP$"),
            ',' => result.push_str("$C$"),

            // '.' doesn't occur in types and functions, so reuse it
            // for ':' and '-'
            '-' | ':' => result.push_char('.'),

            // These are legal symbols
            'a' .. 'z'
            | 'A' .. 'Z'
            | '0' .. '9'
            | '_' | '.' | '$' => result.push_char(c),

            _ => {
                let mut tstr = ~"";
                char::escape_unicode(c, |c| tstr.push_char(c));
                result.push_char('$');
                result.push_str(tstr.slice_from(1));
            }
        }
    }

    // Underscore-qualify anything that didn't start as an ident.
    if result.len() > 0u &&
        result[0] != '_' as u8 &&
        ! char::is_XID_start(result[0] as char) {
        return ~"_" + result;
    }

    return result;
}

pub fn mangle<PI: Iterator<PathElem>>(mut path: PI,
                                      hash: Option<&str>,
                                      vers: Option<&str>) -> ~str {
    // Follow C++ namespace-mangling style, see
    // http://en.wikipedia.org/wiki/Name_mangling for more info.
    //
    // It turns out that on OSX you can actually have arbitrary symbols in
    // function names (at least when given to LLVM), but this is not possible
    // when using unix's linker. Perhaps one day when we just use a linker from LLVM
    // we won't need to do this name mangling. The problem with name mangling is
    // that it seriously limits the available characters. For example we can't
    // have things like &T or ~[T] in symbol names when one would theoretically
    // want them for things like impls of traits on that type.
    //
    // To be able to work on all platforms and get *some* reasonable output, we
    // use C++ name-mangling.

    let mut n = ~"_ZN"; // _Z == Begin name-sequence, N == nested

    fn push(n: &mut ~str, s: &str) {
        let sani = sanitize(s);
        n.push_str(format!("{}{}", sani.len(), sani));
    }

    // First, connect each component with <len, name> pairs.
    for e in path {
        push(&mut n, token::get_name(e.name()).get().as_slice())
    }

    match hash {
        Some(s) => push(&mut n, s),
        None => {}
    }
    match vers {
        Some(s) => push(&mut n, s),
        None => {}
    }

    n.push_char('E'); // End name-sequence.
    n
}

pub fn exported_name(path: PathElems, hash: &str, vers: &str) -> ~str {
    // The version will get mangled to have a leading '_', but it makes more
    // sense to lead with a 'v' b/c this is a version...
    let vers = if vers.len() > 0 && !char::is_XID_start(vers.char_at(0)) {
        "v" + vers
    } else {
        vers.to_owned()
    };

    mangle(path, Some(hash), Some(vers.as_slice()))
}

pub fn mangle_exported_name(ccx: &CrateContext, path: PathElems,
                            t: ty::t, id: ast::NodeId) -> ~str {
    let mut hash = get_symbol_hash(ccx, t);

    // Paths can be completely identical for different nodes,
    // e.g. `fn foo() { { fn a() {} } { fn a() {} } }`, so we
    // generate unique characters from the node id. For now
    // hopefully 3 characters is enough to avoid collisions.
    static EXTRA_CHARS: &'static str =
        "abcdefghijklmnopqrstuvwxyz\
         ABCDEFGHIJKLMNOPQRSTUVWXYZ\
         0123456789";
    let id = id as uint;
    let extra1 = id % EXTRA_CHARS.len();
    let id = id / EXTRA_CHARS.len();
    let extra2 = id % EXTRA_CHARS.len();
    let id = id / EXTRA_CHARS.len();
    let extra3 = id % EXTRA_CHARS.len();
    hash.push_char(EXTRA_CHARS[extra1] as char);
    hash.push_char(EXTRA_CHARS[extra2] as char);
    hash.push_char(EXTRA_CHARS[extra3] as char);

    exported_name(path, hash, ccx.link_meta.crateid.version_or_default())
}

pub fn mangle_internal_name_by_type_and_seq(ccx: &CrateContext,
                                            t: ty::t,
                                            name: &str) -> ~str {
    let s = ppaux::ty_to_str(ccx.tcx(), t);
    let path = [PathName(token::intern(s)),
                gensym_name(name)];
    let hash = get_symbol_hash(ccx, t);
    mangle(ast_map::Values(path.iter()), Some(hash.as_slice()), None)
}

pub fn mangle_internal_name_by_path_and_seq(path: PathElems, flav: &str) -> ~str {
    mangle(path.chain(Some(gensym_name(flav)).move_iter()), None, None)
}

pub fn output_lib_filename(id: &CrateId) -> ~str {
    format!("{}-{}-{}", id.name, crate_id_hash(id), id.version_or_default())
}

pub fn get_cc_prog(sess: &Session) -> ~str {
    match sess.opts.cg.linker {
        Some(ref linker) => return linker.to_owned(),
        None => {}
    }

    // In the future, FreeBSD will use clang as default compiler.
    // It would be flexible to use cc (system's default C compiler)
    // instead of hard-coded gcc.
    // For win32, there is no cc command, so we add a condition to make it use gcc.
    match sess.targ_cfg.os {
        abi::OsWin32 => return ~"gcc",
        _ => {},
    }

    get_system_tool(sess, "cc")
}

pub fn get_ar_prog(sess: &Session) -> ~str {
    match sess.opts.cg.ar {
        Some(ref ar) => return ar.to_owned(),
        None => {}
    }

    get_system_tool(sess, "ar")
}

fn get_system_tool(sess: &Session, tool: &str) -> ~str {
    match sess.targ_cfg.os {
        abi::OsAndroid => match sess.opts.cg.android_cross_path {
            Some(ref path) => {
                let tool_str = match tool {
                    "cc" => "gcc",
                    _ => tool
                };
                format!("{}/bin/arm-linux-androideabi-{}", *path, tool_str)
            }
            None => {
                sess.fatal(format!("need Android NDK path for the '{}' tool \
                                    (-C android-cross-path)", tool))
            }
        },
        _ => tool.to_owned(),
    }
}

fn remove(sess: &Session, path: &Path) {
    match fs::unlink(path) {
        Ok(..) => {}
        Err(e) => {
            sess.err(format!("failed to remove {}: {}", path.display(), e));
        }
    }
}

/// Perform the linkage portion of the compilation phase. This will generate all
/// of the requested outputs for this compilation session.
pub fn link_binary(sess: &Session,
                   trans: &CrateTranslation,
                   outputs: &OutputFilenames,
                   id: &CrateId) -> Vec<Path> {
    let mut out_filenames = Vec::new();
    for &crate_type in sess.crate_types.borrow().iter() {
        let out_file = link_binary_output(sess, trans, crate_type, outputs, id);
        out_filenames.push(out_file);
    }

    // Remove the temporary object file and metadata if we aren't saving temps
    if !sess.opts.cg.save_temps {
        let obj_filename = outputs.temp_path(OutputTypeObject);
        if !sess.opts.output_types.contains(&OutputTypeObject) {
            remove(sess, &obj_filename);
        }
        remove(sess, &obj_filename.with_extension("metadata.o"));
    }

    out_filenames
}

fn is_writeable(p: &Path) -> bool {
    match p.stat() {
        Err(..) => true,
        Ok(m) => m.perm & io::UserWrite == io::UserWrite
    }
}

pub fn filename_for_input(sess: &Session, crate_type: session::CrateType,
                          id: &CrateId, out_filename: &Path) -> Path {
    let libname = output_lib_filename(id);
    match crate_type {
        session::CrateTypeRlib => {
            out_filename.with_filename(format!("lib{}.rlib", libname))
        }
        session::CrateTypeDylib => {
            let (prefix, suffix) = match sess.targ_cfg.os {
                abi::OsWin32 => (win32::DLL_PREFIX, win32::DLL_SUFFIX),
                abi::OsMacos => (macos::DLL_PREFIX, macos::DLL_SUFFIX),
                abi::OsLinux => (linux::DLL_PREFIX, linux::DLL_SUFFIX),
                abi::OsAndroid => (android::DLL_PREFIX, android::DLL_SUFFIX),
                abi::OsFreebsd => (freebsd::DLL_PREFIX, freebsd::DLL_SUFFIX),
            };
            out_filename.with_filename(format!("{}{}{}", prefix, libname, suffix))
        }
        session::CrateTypeStaticlib => {
            out_filename.with_filename(format!("lib{}.a", libname))
        }
        session::CrateTypeExecutable => out_filename.clone(),
    }
}

fn link_binary_output(sess: &Session,
                      trans: &CrateTranslation,
                      crate_type: session::CrateType,
                      outputs: &OutputFilenames,
                      id: &CrateId) -> Path {
    let obj_filename = outputs.temp_path(OutputTypeObject);
    let out_filename = match outputs.single_output_file {
        Some(ref file) => file.clone(),
        None => {
            let out_filename = outputs.path(OutputTypeExe);
            filename_for_input(sess, crate_type, id, &out_filename)
        }
    };

    // Make sure the output and obj_filename are both writeable.
    // Mac, FreeBSD, and Windows system linkers check this already --
    // however, the Linux linker will happily overwrite a read-only file.
    // We should be consistent.
    let obj_is_writeable = is_writeable(&obj_filename);
    let out_is_writeable = is_writeable(&out_filename);
    if !out_is_writeable {
        sess.fatal(format!("output file {} is not writeable -- check its permissions.",
                           out_filename.display()));
    }
    else if !obj_is_writeable {
        sess.fatal(format!("object file {} is not writeable -- check its permissions.",
                           obj_filename.display()));
    }

    match crate_type {
        session::CrateTypeRlib => {
            link_rlib(sess, Some(trans), &obj_filename, &out_filename);
        }
        session::CrateTypeStaticlib => {
            link_staticlib(sess, &obj_filename, &out_filename);
        }
        session::CrateTypeExecutable => {
            link_natively(sess, false, &obj_filename, &out_filename);
        }
        session::CrateTypeDylib => {
            link_natively(sess, true, &obj_filename, &out_filename);
        }
    }

    out_filename
}

// Create an 'rlib'
//
// An rlib in its current incarnation is essentially a renamed .a file. The
// rlib primarily contains the object file of the crate, but it also contains
// all of the object files from native libraries. This is done by unzipping
// native libraries and inserting all of the contents into this archive.
fn link_rlib<'a>(sess: &'a Session,
                 trans: Option<&CrateTranslation>, // None == no metadata/bytecode
                 obj_filename: &Path,
                 out_filename: &Path) -> Archive<'a> {
    let mut a = Archive::create(sess, out_filename, obj_filename);

    for &(ref l, kind) in sess.cstore.get_used_libraries().borrow().iter() {
        match kind {
            cstore::NativeStatic => {
                a.add_native_library(l.as_slice()).unwrap();
            }
            cstore::NativeFramework | cstore::NativeUnknown => {}
        }
    }

    // Note that it is important that we add all of our non-object "magical
    // files" *after* all of the object files in the archive. The reason for
    // this is as follows:
    //
    // * When performing LTO, this archive will be modified to remove
    //   obj_filename from above. The reason for this is described below.
    //
    // * When the system linker looks at an archive, it will attempt to
    //   determine the architecture of the archive in order to see whether its
    //   linkable.
    //
    //   The algorithm for this detection is: iterate over the files in the
    //   archive. Skip magical SYMDEF names. Interpret the first file as an
    //   object file. Read architecture from the object file.
    //
    // * As one can probably see, if "metadata" and "foo.bc" were placed
    //   before all of the objects, then the architecture of this archive would
    //   not be correctly inferred once 'foo.o' is removed.
    //
    // Basically, all this means is that this code should not move above the
    // code above.
    match trans {
        Some(trans) => {
            // Instead of putting the metadata in an object file section, rlibs
            // contain the metadata in a separate file. We use a temp directory
            // here so concurrent builds in the same directory don't try to use
            // the same filename for metadata (stomping over one another)
            let tmpdir = TempDir::new("rustc").expect("needs a temp dir");
            let metadata = tmpdir.path().join(METADATA_FILENAME);
            match fs::File::create(&metadata).write(trans.metadata
                                                         .as_slice()) {
                Ok(..) => {}
                Err(e) => {
                    sess.err(format!("failed to write {}: {}",
                                     metadata.display(), e));
                    sess.abort_if_errors();
                }
            }
            a.add_file(&metadata, false);
            remove(sess, &metadata);

            // For LTO purposes, the bytecode of this library is also inserted
            // into the archive.
            let bc = obj_filename.with_extension("bc");
            let bc_deflated = obj_filename.with_extension("bc.deflate");
            match fs::File::open(&bc).read_to_end().and_then(|data| {
                fs::File::create(&bc_deflated)
                    .write(match flate::deflate_bytes(data.as_slice()) {
                        Some(compressed) => compressed,
                        None => sess.fatal("failed to compress bytecode")
                     }.as_slice())
            }) {
                Ok(()) => {}
                Err(e) => {
                    sess.err(format!("failed to write compressed bytecode: {}", e));
                    sess.abort_if_errors()
                }
            }
            a.add_file(&bc_deflated, false);
            remove(sess, &bc_deflated);
            if !sess.opts.cg.save_temps &&
               !sess.opts.output_types.contains(&OutputTypeBitcode) {
                remove(sess, &bc);
            }

            // After adding all files to the archive, we need to update the
            // symbol table of the archive. This currently dies on OSX (see
            // #11162), and isn't necessary there anyway
            match sess.targ_cfg.os {
                abi::OsMacos => {}
                _ => { a.update_symbols(); }
            }
        }

        None => {}
    }
    return a;
}

// Create a static archive
//
// This is essentially the same thing as an rlib, but it also involves adding
// all of the upstream crates' objects into the archive. This will slurp in
// all of the native libraries of upstream dependencies as well.
//
// Additionally, there's no way for us to link dynamic libraries, so we warn
// about all dynamic library dependencies that they're not linked in.
//
// There's no need to include metadata in a static archive, so ensure to not
// link in the metadata object file (and also don't prepare the archive with a
// metadata file).
fn link_staticlib(sess: &Session, obj_filename: &Path, out_filename: &Path) {
    let mut a = link_rlib(sess, None, obj_filename, out_filename);
    a.add_native_library("morestack").unwrap();
    a.add_native_library("compiler-rt").unwrap();

    let crates = sess.cstore.get_used_crates(cstore::RequireStatic);
    for &(cnum, ref path) in crates.iter() {
        let name = sess.cstore.get_crate_data(cnum).name.clone();
        let p = match *path {
            Some(ref p) => p.clone(), None => {
                sess.err(format!("could not find rlib for: `{}`", name));
                continue
            }
        };
        a.add_rlib(&p, name, sess.lto()).unwrap();
        let native_libs = csearch::get_native_libraries(&sess.cstore, cnum);
        for &(kind, ref lib) in native_libs.iter() {
            let name = match kind {
                cstore::NativeStatic => "static library",
                cstore::NativeUnknown => "library",
                cstore::NativeFramework => "framework",
            };
            sess.warn(format!("unlinked native {}: {}", name, *lib));
        }
    }
}

// Create a dynamic library or executable
//
// This will invoke the system linker/cc to create the resulting file. This
// links to all upstream files as well.
fn link_natively(sess: &Session, dylib: bool, obj_filename: &Path,
                 out_filename: &Path) {
    let tmpdir = TempDir::new("rustc").expect("needs a temp dir");
    // The invocations of cc share some flags across platforms
    let cc_prog = get_cc_prog(sess);
    let mut cc_args = sess.targ_cfg.target_strs.cc_args.clone();
    cc_args.push_all_move(link_args(sess, dylib, tmpdir.path(),
                                    obj_filename, out_filename));
    if (sess.opts.debugging_opts & session::PRINT_LINK_ARGS) != 0 {
        println!("{} link args: '{}'", cc_prog, cc_args.connect("' '"));
    }

    // May have not found libraries in the right formats.
    sess.abort_if_errors();

    // Invoke the system linker
    debug!("{} {}", cc_prog, cc_args.connect(" "));
    let prog = time(sess.time_passes(), "running linker", (), |()|
                    Process::output(cc_prog, cc_args.as_slice()));
    match prog {
        Ok(prog) => {
            if !prog.status.success() {
                sess.err(format!("linking with `{}` failed: {}", cc_prog, prog.status));
                sess.note(format!("{} arguments: '{}'", cc_prog, cc_args.connect("' '")));
                let mut output = prog.error.clone();
                output.push_all(prog.output.as_slice());
                sess.note(str::from_utf8(output.as_slice()).unwrap().to_owned());
                sess.abort_if_errors();
            }
        },
        Err(e) => {
            sess.err(format!("could not exec the linker `{}`: {}", cc_prog, e));
            sess.abort_if_errors();
        }
    }


    // On OSX, debuggers need this utility to get run to do some munging of
    // the symbols
    if sess.targ_cfg.os == abi::OsMacos && (sess.opts.debuginfo != NoDebugInfo) {
        // FIXME (#9639): This needs to handle non-utf8 paths
        match Process::status("dsymutil",
                                  [out_filename.as_str().unwrap().to_owned()]) {
            Ok(..) => {}
            Err(e) => {
                sess.err(format!("failed to run dsymutil: {}", e));
                sess.abort_if_errors();
            }
        }
    }
}

fn link_args(sess: &Session,
             dylib: bool,
             tmpdir: &Path,
             obj_filename: &Path,
             out_filename: &Path) -> Vec<~str> {

    // The default library location, we need this to find the runtime.
    // The location of crates will be determined as needed.
    // FIXME (#9639): This needs to handle non-utf8 paths
    let lib_path = sess.filesearch().get_target_lib_path();
    let stage: ~str = ~"-L" + lib_path.as_str().unwrap();

    let mut args = vec!(stage);

    // FIXME (#9639): This needs to handle non-utf8 paths
    args.push_all([
        ~"-o", out_filename.as_str().unwrap().to_owned(),
        obj_filename.as_str().unwrap().to_owned()]);

    // Stack growth requires statically linking a __morestack function. Note
    // that this is listed *before* all other libraries, even though it may be
    // used to resolve symbols in other libraries. The only case that this
    // wouldn't be pulled in by the object file is if the object file had no
    // functions.
    //
    // If we're building an executable, there must be at least one function (the
    // main function), and if we're building a dylib then we don't need it for
    // later libraries because they're all dylibs (not rlibs).
    //
    // I'm honestly not entirely sure why this needs to come first. Apparently
    // the --as-needed flag above sometimes strips out libstd from the command
    // line, but inserting this farther to the left makes the
    // "rust_stack_exhausted" symbol an outstanding undefined symbol, which
    // flags libstd as a required library (or whatever provides the symbol).
    args.push(~"-lmorestack");

    // When linking a dynamic library, we put the metadata into a section of the
    // executable. This metadata is in a separate object file from the main
    // object file, so we link that in here.
    if dylib {
        let metadata = obj_filename.with_extension("metadata.o");
        args.push(metadata.as_str().unwrap().to_owned());
    }

    // We want to prevent the compiler from accidentally leaking in any system
    // libraries, so we explicitly ask gcc to not link to any libraries by
    // default. Note that this does not happen for windows because windows pulls
    // in some large number of libraries and I couldn't quite figure out which
    // subset we wanted.
    //
    // FIXME(#11937) we should invoke the system linker directly
    if sess.targ_cfg.os != abi::OsWin32 {
        args.push(~"-nodefaultlibs");
    }

    if sess.targ_cfg.os == abi::OsLinux {
        // GNU-style linkers will use this to omit linking to libraries which
        // don't actually fulfill any relocations, but only for libraries which
        // follow this flag. Thus, use it before specifying libraries to link to.
        args.push(~"-Wl,--as-needed");

        // GNU-style linkers support optimization with -O. --gc-sections
        // removes metadata and potentially other useful things, so don't
        // include it. GNU ld doesn't need a numeric argument, but other linkers
        // do.
        if sess.opts.optimize == session::Default ||
           sess.opts.optimize == session::Aggressive {
            args.push(~"-Wl,-O1");
        }
    }

    if sess.targ_cfg.os == abi::OsWin32 {
        // Make sure that we link to the dynamic libgcc, otherwise cross-module
        // DWARF stack unwinding will not work.
        // This behavior may be overridden by --link-args "-static-libgcc"
        args.push(~"-shared-libgcc");

        // And here, we see obscure linker flags #45. On windows, it has been
        // found to be necessary to have this flag to compile liblibc.
        //
        // First a bit of background. On Windows, the file format is not ELF,
        // but COFF (at least according to LLVM). COFF doesn't officially allow
        // for section names over 8 characters, apparently. Our metadata
        // section, ".note.rustc", you'll note is over 8 characters.
        //
        // On more recent versions of gcc on mingw, apparently the section name
        // is *not* truncated, but rather stored elsewhere in a separate lookup
        // table. On older versions of gcc, they apparently always truncated the
        // section names (at least in some cases). Truncating the section name
        // actually creates "invalid" objects [1] [2], but only for some
        // introspection tools, not in terms of whether it can be loaded.
        //
        // Long story shory, passing this flag forces the linker to *not*
        // truncate section names (so we can find the metadata section after
        // it's compiled). The real kicker is that rust compiled just fine on
        // windows for quite a long time *without* this flag, so I have no idea
        // why it suddenly started failing for liblibc. Regardless, we
        // definitely don't want section name truncation, so we're keeping this
        // flag for windows.
        //
        // [1] - https://sourceware.org/bugzilla/show_bug.cgi?id=13130
        // [2] - https://code.google.com/p/go/issues/detail?id=2139
        args.push(~"-Wl,--enable-long-section-names");
    }

    if sess.targ_cfg.os == abi::OsAndroid {
        // Many of the symbols defined in compiler-rt are also defined in libgcc.
        // Android linker doesn't like that by default.
        args.push(~"-Wl,--allow-multiple-definition");
    }

    // Take careful note of the ordering of the arguments we pass to the linker
    // here. Linkers will assume that things on the left depend on things to the
    // right. Things on the right cannot depend on things on the left. This is
    // all formally implemented in terms of resolving symbols (libs on the right
    // resolve unknown symbols of libs on the left, but not vice versa).
    //
    // For this reason, we have organized the arguments we pass to the linker as
    // such:
    //
    //  1. The local object that LLVM just generated
    //  2. Upstream rust libraries
    //  3. Local native libraries
    //  4. Upstream native libraries
    //
    // This is generally fairly natural, but some may expect 2 and 3 to be
    // swapped. The reason that all native libraries are put last is that it's
    // not recommended for a native library to depend on a symbol from a rust
    // crate. If this is the case then a staticlib crate is recommended, solving
    // the problem.
    //
    // Additionally, it is occasionally the case that upstream rust libraries
    // depend on a local native library. In the case of libraries such as
    // lua/glfw/etc the name of the library isn't the same across all platforms,
    // so only the consumer crate of a library knows the actual name. This means
    // that downstream crates will provide the #[link] attribute which upstream
    // crates will depend on. Hence local native libraries are after out
    // upstream rust crates.
    //
    // In theory this means that a symbol in an upstream native library will be
    // shadowed by a local native library when it wouldn't have been before, but
    // this kind of behavior is pretty platform specific and generally not
    // recommended anyway, so I don't think we're shooting ourself in the foot
    // much with that.
    add_upstream_rust_crates(&mut args, sess, dylib, tmpdir);
    add_local_native_libraries(&mut args, sess);
    add_upstream_native_libraries(&mut args, sess);

    // # Telling the linker what we're doing

    if dylib {
        // On mac we need to tell the linker to let this library be rpathed
        if sess.targ_cfg.os == abi::OsMacos {
            args.push(~"-dynamiclib");
            args.push(~"-Wl,-dylib");
            // FIXME (#9639): This needs to handle non-utf8 paths
            if !sess.opts.cg.no_rpath {
                args.push(~"-Wl,-install_name,@rpath/" +
                          out_filename.filename_str().unwrap());
            }
        } else {
            args.push(~"-shared")
        }
    }

    if sess.targ_cfg.os == abi::OsFreebsd {
        args.push_all([~"-L/usr/local/lib",
                       ~"-L/usr/local/lib/gcc46",
                       ~"-L/usr/local/lib/gcc44"]);
    }

    // FIXME (#2397): At some point we want to rpath our guesses as to
    // where extern libraries might live, based on the
    // addl_lib_search_paths
    if !sess.opts.cg.no_rpath {
        args.push_all(rpath::get_rpath_flags(sess, out_filename).as_slice());
    }

    // compiler-rt contains implementations of low-level LLVM helpers. This is
    // used to resolve symbols from the object file we just created, as well as
    // any system static libraries that may be expecting gcc instead. Most
    // symbols in libgcc also appear in compiler-rt.
    //
    // This is the end of the command line, so this library is used to resolve
    // *all* undefined symbols in all other libraries, and this is intentional.
    args.push(~"-lcompiler-rt");

    // Finally add all the linker arguments provided on the command line along
    // with any #[link_args] attributes found inside the crate
    args.push_all(sess.opts.cg.link_args.as_slice());
    for arg in sess.cstore.get_used_link_args().borrow().iter() {
        args.push(arg.clone());
    }
    return args;
}

// # Native library linking
//
// User-supplied library search paths (-L on the command line). These are
// the same paths used to find Rust crates, so some of them may have been
// added already by the previous crate linking code. This only allows them
// to be found at compile time so it is still entirely up to outside
// forces to make sure that library can be found at runtime.
//
// Also note that the native libraries linked here are only the ones located
// in the current crate. Upstream crates with native library dependencies
// may have their native library pulled in above.
fn add_local_native_libraries(args: &mut Vec<~str>, sess: &Session) {
    for path in sess.opts.addl_lib_search_paths.borrow().iter() {
        // FIXME (#9639): This needs to handle non-utf8 paths
        args.push("-L" + path.as_str().unwrap().to_owned());
    }

    let rustpath = filesearch::rust_path();
    for path in rustpath.iter() {
        // FIXME (#9639): This needs to handle non-utf8 paths
        args.push("-L" + path.as_str().unwrap().to_owned());
    }

    for &(ref l, kind) in sess.cstore.get_used_libraries().borrow().iter() {
        match kind {
            cstore::NativeUnknown | cstore::NativeStatic => {
                args.push("-l" + *l);
            }
            cstore::NativeFramework => {
                args.push(~"-framework");
                args.push(l.to_owned());
            }
        }
    }
}

// # Rust Crate linking
//
// Rust crates are not considered at all when creating an rlib output. All
// dependencies will be linked when producing the final output (instead of
// the intermediate rlib version)
fn add_upstream_rust_crates(args: &mut Vec<~str>, sess: &Session,
                            dylib: bool, tmpdir: &Path) {

    // As a limitation of the current implementation, we require that everything
    // must be static or everything must be dynamic. The reasons for this are a
    // little subtle, but as with staticlibs and rlibs, the goal is to prevent
    // duplicate copies of the same library showing up. For example, a static
    // immediate dependency might show up as an upstream dynamic dependency and
    // we currently have no way of knowing that. We know that all dynamic
    // libraries require dynamic dependencies (see above), so it's satisfactory
    // to include either all static libraries or all dynamic libraries.
    //
    // With this limitation, we expose a compiler default linkage type and an
    // option to reverse that preference. The current behavior looks like:
    //
    // * If a dylib is being created, upstream dependencies must be dylibs
    // * If nothing else is specified, static linking is preferred
    // * If the -C prefer-dynamic flag is given, dynamic linking is preferred
    // * If one form of linking fails, the second is also attempted
    // * If both forms fail, then we emit an error message

    let dynamic = get_deps(&sess.cstore, cstore::RequireDynamic);
    let statik = get_deps(&sess.cstore, cstore::RequireStatic);
    match (dynamic, statik, sess.opts.cg.prefer_dynamic, dylib) {
        (_, Some(deps), false, false) => {
            add_static_crates(args, sess, tmpdir, deps)
        }

        (None, Some(deps), true, false) => {
            // If you opted in to dynamic linking and we decided to emit a
            // static output, you should probably be notified of such an event!
            sess.warn("dynamic linking was preferred, but dependencies \
                       could not all be found in an dylib format.");
            sess.warn("linking statically instead, using rlibs");
            add_static_crates(args, sess, tmpdir, deps)
        }

        (Some(deps), _, _, _) => add_dynamic_crates(args, sess, deps),

        (None, _, _, true) => {
            sess.err("dylib output requested, but some depenencies could not \
                      be found in the dylib format");
            let deps = sess.cstore.get_used_crates(cstore::RequireDynamic);
            for (cnum, path) in deps.move_iter() {
                if path.is_some() { continue }
                let name = sess.cstore.get_crate_data(cnum).name.clone();
                sess.note(format!("dylib not found: {}", name));
            }
        }

        (None, None, pref, false) => {
            let (pref, name) = if pref {
                sess.err("dynamic linking is preferred, but dependencies were \
                          not found in either dylib or rlib format");
                (cstore::RequireDynamic, "dylib")
            } else {
                sess.err("dependencies were not all found in either dylib or \
                          rlib format");
                (cstore::RequireStatic, "rlib")
            };
            sess.note(format!("dependencies not found in the `{}` format",
                              name));
            for (cnum, path) in sess.cstore.get_used_crates(pref).move_iter() {
                if path.is_some() { continue }
                let name = sess.cstore.get_crate_data(cnum).name.clone();
                sess.note(name);
            }
        }
    }

    // Converts a library file-stem into a cc -l argument
    fn unlib(config: &session::Config, stem: &str) -> ~str {
        if stem.starts_with("lib") && config.os != abi::OsWin32 {
            stem.slice(3, stem.len()).to_owned()
        } else {
            stem.to_owned()
        }
    }

    // Attempts to find all dependencies with a certain linkage preference,
    // returning `None` if not all libraries could be found with that
    // preference.
    fn get_deps(cstore: &cstore::CStore,  preference: cstore::LinkagePreference)
            -> Option<Vec<(ast::CrateNum, Path)> >
    {
        let crates = cstore.get_used_crates(preference);
        if crates.iter().all(|&(_, ref p)| p.is_some()) {
            Some(crates.move_iter().map(|(a, b)| (a, b.unwrap())).collect())
        } else {
            None
        }
    }

    // Adds the static "rlib" versions of all crates to the command line.
    fn add_static_crates(args: &mut Vec<~str>, sess: &Session, tmpdir: &Path,
                         crates: Vec<(ast::CrateNum, Path)>) {
        for (cnum, cratepath) in crates.move_iter() {
            // When performing LTO on an executable output, all of the
            // bytecode from the upstream libraries has already been
            // included in our object file output. We need to modify all of
            // the upstream archives to remove their corresponding object
            // file to make sure we don't pull the same code in twice.
            //
            // We must continue to link to the upstream archives to be sure
            // to pull in native static dependencies. As the final caveat,
            // on linux it is apparently illegal to link to a blank archive,
            // so if an archive no longer has any object files in it after
            // we remove `lib.o`, then don't link against it at all.
            //
            // If we're not doing LTO, then our job is simply to just link
            // against the archive.
            if sess.lto() {
                let name = sess.cstore.get_crate_data(cnum).name.clone();
                time(sess.time_passes(), format!("altering {}.rlib", name),
                     (), |()| {
                    let dst = tmpdir.join(cratepath.filename().unwrap());
                    match fs::copy(&cratepath, &dst) {
                        Ok(..) => {}
                        Err(e) => {
                            sess.err(format!("failed to copy {} to {}: {}",
                                             cratepath.display(),
                                             dst.display(),
                                             e));
                            sess.abort_if_errors();
                        }
                    }
                    let dst_str = dst.as_str().unwrap().to_owned();
                    let mut archive = Archive::open(sess, dst);
                    archive.remove_file(format!("{}.o", name));
                    let files = archive.files();
                    if files.iter().any(|s| s.ends_with(".o")) {
                        args.push(dst_str);
                    }
                });
            } else {
                args.push(cratepath.as_str().unwrap().to_owned());
            }
        }
    }

    // Same thing as above, but for dynamic crates instead of static crates.
    fn add_dynamic_crates(args: &mut Vec<~str>, sess: &Session,
                          crates: Vec<(ast::CrateNum, Path)> ) {
        // If we're performing LTO, then it should have been previously required
        // that all upstream rust dependencies were available in an rlib format.
        assert!(!sess.lto());

        for (_, cratepath) in crates.move_iter() {
            // Just need to tell the linker about where the library lives and
            // what its name is
            let dir = cratepath.dirname_str().unwrap();
            if !dir.is_empty() { args.push("-L" + dir); }
            let libarg = unlib(&sess.targ_cfg, cratepath.filestem_str().unwrap());
            args.push("-l" + libarg);
        }
    }
}

// Link in all of our upstream crates' native dependencies. Remember that
// all of these upstream native depenencies are all non-static
// dependencies. We've got two cases then:
//
// 1. The upstream crate is an rlib. In this case we *must* link in the
//    native dependency because the rlib is just an archive.
//
// 2. The upstream crate is a dylib. In order to use the dylib, we have to
//    have the dependency present on the system somewhere. Thus, we don't
//    gain a whole lot from not linking in the dynamic dependency to this
//    crate as well.
//
// The use case for this is a little subtle. In theory the native
// dependencies of a crate a purely an implementation detail of the crate
// itself, but the problem arises with generic and inlined functions. If a
// generic function calls a native function, then the generic function must
// be instantiated in the target crate, meaning that the native symbol must
// also be resolved in the target crate.
fn add_upstream_native_libraries(args: &mut Vec<~str>, sess: &Session) {
    let cstore = &sess.cstore;
    cstore.iter_crate_data(|cnum, _| {
        let libs = csearch::get_native_libraries(cstore, cnum);
        for &(kind, ref lib) in libs.iter() {
            match kind {
                cstore::NativeUnknown => args.push("-l" + *lib),
                cstore::NativeFramework => {
                    args.push(~"-framework");
                    args.push(lib.to_owned());
                }
                cstore::NativeStatic => {
                    sess.bug("statics shouldn't be propagated");
                }
            }
        }
    });
}
