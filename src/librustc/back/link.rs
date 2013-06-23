// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use back::rpath;
use driver::session::Session;
use driver::session;
use lib::llvm::llvm;
use lib::llvm::ModuleRef;
use lib;
use metadata::common::LinkMeta;
use metadata::{encoder, csearch, cstore};
use middle::trans::context::CrateContext;
use middle::ty;
use util::ppaux;

use core::char;
use core::hash::Streaming;
use core::hash;
use core::libc::{c_int, c_uint};
use core::os::consts::{macos, freebsd, linux, android, win32};
use core::os;
use core::ptr;
use core::rt::io::Writer;
use core::run;
use core::str;
use core::vec;
use syntax::ast;
use syntax::ast_map::{path, path_mod, path_name};
use syntax::attr;
use syntax::print::pprust;

#[deriving(Eq)]
pub enum output_type {
    output_type_none,
    output_type_bitcode,
    output_type_assembly,
    output_type_llvm_assembly,
    output_type_object,
    output_type_exe,
}

fn write_string<W:Writer>(writer: &mut W, string: &str) {
    writer.write(string.as_bytes());
}

pub fn llvm_err(sess: Session, msg: ~str) -> ! {
    unsafe {
        let cstr = llvm::LLVMRustGetLastError();
        if cstr == ptr::null() {
            sess.fatal(msg);
        } else {
            sess.fatal(msg + ": " + str::raw::from_c_str(cstr));
        }
    }
}

pub fn WriteOutputFile(sess: Session,
        PM: lib::llvm::PassManagerRef, M: ModuleRef,
        Triple: &str,
        Feature: &str,
        Output: &str,
        // FIXME: When #2334 is fixed, change
        // c_uint to FileType
        FileType: c_uint,
        OptLevel: c_int,
        EnableSegmentedStacks: bool) {
    unsafe {
        do str::as_c_str(Triple) |Triple| {
            do str::as_c_str(Feature) |Feature| {
                do str::as_c_str(Output) |Output| {
                    let result = llvm::LLVMRustWriteOutputFile(
                            PM,
                            M,
                            Triple,
                            Feature,
                            Output,
                            FileType,
                            OptLevel,
                            EnableSegmentedStacks);
                    if (!result) {
                        llvm_err(sess, ~"Could not write output");
                    }
                }
            }
        }
    }
}

pub mod jit {
    use core::prelude::*;

    use back::link::llvm_err;
    use driver::session::Session;
    use lib::llvm::llvm;
    use lib::llvm::{ModuleRef, ContextRef};
    use metadata::cstore;

    use core::cast;
    use core::ptr;
    use core::str;
    use core::sys;
    use core::unstable::intrinsics;

    pub fn exec(sess: Session,
                c: ContextRef,
                m: ModuleRef,
                stacks: bool) {
        unsafe {
            let manager = llvm::LLVMRustPrepareJIT(intrinsics::morestack_addr());

            // We need to tell JIT where to resolve all linked
            // symbols from. The equivalent of -lstd, -lcore, etc.
            // By default the JIT will resolve symbols from the extra and
            // core linked into rustc. We don't want that,
            // incase the user wants to use an older extra library.

            let cstore = sess.cstore;
            let r = cstore::get_used_crate_files(cstore);
            for r.iter().advance |cratepath| {
                let path = cratepath.to_str();

                debug!("linking: %s", path);

                do str::as_c_str(path) |buf_t| {
                    if !llvm::LLVMRustLoadCrate(manager, buf_t) {
                        llvm_err(sess, ~"Could not link");
                    }
                    debug!("linked: %s", path);
                }
            }

            // We custom-build a JIT execution engine via some rust wrappers
            // first. This wrappers takes ownership of the module passed in.
            let ee = llvm::LLVMRustBuildJIT(manager, m, stacks);
            if ee.is_null() {
                llvm::LLVMContextDispose(c);
                llvm_err(sess, ~"Could not create the JIT");
            }

            // Next, we need to get a handle on the _rust_main function by
            // looking up it's corresponding ValueRef and then requesting that
            // the execution engine compiles the function.
            let fun = do str::as_c_str("_rust_main") |entry| {
                llvm::LLVMGetNamedFunction(m, entry)
            };
            if fun.is_null() {
                llvm::LLVMDisposeExecutionEngine(ee);
                llvm::LLVMContextDispose(c);
                llvm_err(sess, ~"Could not find _rust_main in the JIT");
            }

            // Finally, once we have the pointer to the code, we can do some
            // closure magic here to turn it straight into a callable rust
            // closure
            let code = llvm::LLVMGetPointerToGlobal(ee, fun);
            assert!(!code.is_null());
            let closure = sys::Closure {
                code: code,
                env: ptr::null()
            };
            let func: &fn() = cast::transmute(closure);
            func();

            // Sadly, there currently is no interface to re-use this execution
            // engine, so it's disposed of here along with the context to
            // prevent leaks.
            llvm::LLVMDisposeExecutionEngine(ee);
            llvm::LLVMContextDispose(c);
        }
    }
}

pub mod write {
    use core::prelude::*;

    use back::link::jit;
    use back::link::{WriteOutputFile, output_type};
    use back::link::{output_type_assembly, output_type_bitcode};
    use back::link::{output_type_exe, output_type_llvm_assembly};
    use back::link::{output_type_object};
    use driver::session::Session;
    use driver::session;
    use lib::llvm::llvm;
    use lib::llvm::{ModuleRef, mk_pass_manager, mk_target_data};
    use lib::llvm::{ContextRef};
    use lib;

    use back::passes;

    use core::libc::{c_int, c_uint};
    use core::path::Path;
    use core::run;
    use core::str;

    pub fn is_object_or_assembly_or_exe(ot: output_type) -> bool {
        match ot {
            output_type_assembly | output_type_object | output_type_exe => true,
            _ => false
        }
    }

    pub fn run_passes(sess: Session,
                      llcx: ContextRef,
                      llmod: ModuleRef,
                      output_type: output_type,
                      output: &Path) {
        unsafe {
            llvm::LLVMInitializePasses();

            let opts = sess.opts;
            if sess.time_llvm_passes() { llvm::LLVMRustEnableTimePasses(); }
            let td = mk_target_data(sess.targ_cfg.target_strs.data_layout);
            let pm = mk_pass_manager();
            llvm::LLVMAddTargetData(td.lltd, pm.llpm);

            // Generate a pre-optimization intermediate file if -save-temps
            // was specified.
            if opts.save_temps {
                match output_type {
                  output_type_bitcode => {
                    if opts.optimize != session::No {
                        let filename = output.with_filetype("no-opt.bc");
                        str::as_c_str(filename.to_str(), |buf| {
                            llvm::LLVMWriteBitcodeToFile(llmod, buf)
                        });
                    }
                  }
                  _ => {
                    let filename = output.with_filetype("bc");
                    str::as_c_str(filename.to_str(), |buf| {
                        llvm::LLVMWriteBitcodeToFile(llmod, buf)
                    });
                  }
                }
            }

            let mut mpm = passes::PassManager::new(td.lltd);

            if !sess.no_verify() {
                mpm.add_pass_from_name("verify");
            }

            let passes = if sess.opts.custom_passes.len() > 0 {
                copy sess.opts.custom_passes
            } else {
                if sess.lint_llvm() {
                    mpm.add_pass_from_name("lint");
                }
                passes::create_standard_passes(opts.optimize)
            };


            debug!("Passes: %?", passes);
            passes::populate_pass_manager(sess, &mut mpm, passes);

            debug!("Running Module Optimization Pass");
            mpm.run(llmod);

            if opts.jit {
                // If we are using JIT, go ahead and create and execute the
                // engine now.  JIT execution takes ownership of the module and
                // context, so don't dispose and return.
                jit::exec(sess, llcx, llmod, true);

                if sess.time_llvm_passes() {
                    llvm::LLVMRustPrintPassTimings();
                }
                return;
            } else if is_object_or_assembly_or_exe(output_type) {
                let LLVMOptNone       = 0 as c_int; // -O0
                let LLVMOptLess       = 1 as c_int; // -O1
                let LLVMOptDefault    = 2 as c_int; // -O2, -Os
                let LLVMOptAggressive = 3 as c_int; // -O3

                let CodeGenOptLevel = match opts.optimize {
                  session::No => LLVMOptNone,
                  session::Less => LLVMOptLess,
                  session::Default => LLVMOptDefault,
                  session::Aggressive => LLVMOptAggressive
                };

                let FileType = match output_type {
                    output_type_object | output_type_exe => lib::llvm::ObjectFile,
                    _ => lib::llvm::AssemblyFile
                };

                // Write optimized bitcode if --save-temps was on.

                if opts.save_temps {
                    // Always output the bitcode file with --save-temps

                    let filename = output.with_filetype("opt.bc");
                    str::as_c_str(filename.to_str(), |buf| {
                        llvm::LLVMWriteBitcodeToFile(llmod, buf)
                    });
                    // Save the assembly file if -S is used
                    if output_type == output_type_assembly {
                        WriteOutputFile(
                            sess,
                            pm.llpm,
                            llmod,
                            sess.targ_cfg.target_strs.target_triple,
                            opts.target_feature,
                            output.to_str(),
                            lib::llvm::AssemblyFile as c_uint,
                            CodeGenOptLevel,
                            true);
                    }

                    // Save the object file for -c or --save-temps alone
                    // This .o is needed when an exe is built
                    if output_type == output_type_object ||
                           output_type == output_type_exe {
                        WriteOutputFile(
                            sess,
                            pm.llpm,
                            llmod,
                            sess.targ_cfg.target_strs.target_triple,
                            opts.target_feature,
                            output.to_str(),
                            lib::llvm::ObjectFile as c_uint,
                            CodeGenOptLevel,
                            true);
                    }
                } else {
                    // If we aren't saving temps then just output the file
                    // type corresponding to the '-c' or '-S' flag used
                    WriteOutputFile(
                        sess,
                        pm.llpm,
                        llmod,
                        sess.targ_cfg.target_strs.target_triple,
                        opts.target_feature,
                        output.to_str(),
                        FileType as c_uint,
                        CodeGenOptLevel,
                        true);
                }
                // Clean up and return

                llvm::LLVMDisposeModule(llmod);
                llvm::LLVMContextDispose(llcx);
                if sess.time_llvm_passes() {
                    llvm::LLVMRustPrintPassTimings();
                }
                return;
            }

            if output_type == output_type_llvm_assembly {
                // Given options "-S --emit-llvm": output LLVM assembly
                str::as_c_str(output.to_str(), |buf_o| {
                    llvm::LLVMRustAddPrintModulePass(pm.llpm, llmod, buf_o)});
            } else {
                // If only a bitcode file is asked for by using the
                // '--emit-llvm' flag, then output it here
                str::as_c_str(output.to_str(),
                            |buf| llvm::LLVMWriteBitcodeToFile(llmod, buf) );
            }

            llvm::LLVMDisposeModule(llmod);
            llvm::LLVMContextDispose(llcx);
            if sess.time_llvm_passes() { llvm::LLVMRustPrintPassTimings(); }
        }
    }

    pub fn run_ndk(sess: Session, assembly: &Path, object: &Path) {
        let cc_prog: ~str = match &sess.opts.android_cross_path {
            &Some(ref path) => {
                fmt!("%s/bin/arm-linux-androideabi-gcc", *path)
            }
            &None => {
                sess.fatal("need Android NDK path for building \
                            (--android-cross-path)")
            }
        };

        let cc_args = ~[
            ~"-c",
            ~"-o", object.to_str(),
            assembly.to_str()];

        let prog = run::process_output(cc_prog, cc_args);

        if prog.status != 0 {
            sess.err(fmt!("building with `%s` failed with code %d",
                        cc_prog, prog.status));
            sess.note(fmt!("%s arguments: %s",
                        cc_prog, cc_args.connect(" ")));
            sess.note(str::from_bytes(prog.error + prog.output));
            sess.abort_if_errors();
        }
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
 *    we use SHA1) to "prevent collisions". This is not airtight but 16 hex
 *    digits on uniform probability means you're going to need 2**32 same-name
 *    symbols in the same process before you're even hitting birthday-paradox
 *    collision probability.
 *
 *  - Symbols in different crates but with same names "within" the crate need
 *    to get different linkage-names.
 *
 * So here is what we do:
 *
 *  - Separate the meta tags into two sets: exported and local. Only work with
 *    the exported ones when considering linkage.
 *
 *  - Consider two exported tags as special (and mandatory): name and vers.
 *    Every crate gets them; if it doesn't name them explicitly we infer them
 *    as basename(crate) and "0.1", respectively. Call these CNAME, CVERS.
 *
 *  - Define CMETA as all the non-name, non-vers exported meta tags in the
 *    crate (in sorted order).
 *
 *  - Define CMH as hash(CMETA + hashes of dependent crates).
 *
 *  - Compile our crate to lib CNAME-CMH-CVERS.so
 *
 *  - Define STH(sym) as hash(CNAME, CMH, type_str(sym))
 *
 *  - Suffix a mangled sym with ::STH@CVERS, so that it is unique in the
 *    name, non-name metadata, and type sense, and versioned in the way
 *    system linkers understand.
 *
 */

pub fn build_link_meta(sess: Session,
                       c: &ast::crate,
                       output: &Path,
                       symbol_hasher: &mut hash::State)
                       -> LinkMeta {
    struct ProvidedMetas {
        name: Option<@str>,
        vers: Option<@str>,
        cmh_items: ~[@ast::meta_item]
    }

    fn provided_link_metas(sess: Session, c: &ast::crate) ->
       ProvidedMetas {
        let mut name = None;
        let mut vers = None;
        let mut cmh_items = ~[];
        let linkage_metas = attr::find_linkage_metas(c.node.attrs);
        attr::require_unique_names(sess.diagnostic(), linkage_metas);
        for linkage_metas.iter().advance |meta| {
            match attr::get_meta_item_value_str(*meta) {
                Some(value) => {
                    let item_name : &str = attr::get_meta_item_name(*meta);
                    match item_name {
                        // Changing attr would avoid the need for the copy
                        // here
                        "name" => name = Some(value),
                        "vers" => vers = Some(value),
                        _ => cmh_items.push(*meta)
                    }
                },
                None => cmh_items.push(*meta)
            }
        }

        ProvidedMetas {
            name: name,
            vers: vers,
            cmh_items: cmh_items
        }
    }

    // This calculates CMH as defined above
    fn crate_meta_extras_hash(symbol_hasher: &mut hash::State,
                              cmh_items: ~[@ast::meta_item],
                              dep_hashes: ~[@str]) -> @str {
        fn len_and_str(s: &str) -> ~str {
            fmt!("%u_%s", s.len(), s)
        }

        fn len_and_str_lit(l: ast::lit) -> ~str {
            len_and_str(pprust::lit_to_str(@l))
        }

        let cmh_items = attr::sort_meta_items(cmh_items);

        fn hash(symbol_hasher: &mut hash::State, m: &@ast::meta_item) {
            match m.node {
              ast::meta_name_value(key, value) => {
                write_string(symbol_hasher, len_and_str(key));
                write_string(symbol_hasher, len_and_str_lit(value));
              }
              ast::meta_word(name) => {
                write_string(symbol_hasher, len_and_str(name));
              }
              ast::meta_list(name, ref mis) => {
                write_string(symbol_hasher, len_and_str(name));
                for mis.iter().advance |m_| {
                    hash(symbol_hasher, m_);
                }
              }
            }
        }

        symbol_hasher.reset();
        for cmh_items.iter().advance |m| {
            hash(symbol_hasher, m);
        }

        for dep_hashes.iter().advance |dh| {
            write_string(symbol_hasher, len_and_str(*dh));
        }

    // tjc: allocation is unfortunate; need to change core::hash
        return truncated_hash_result(symbol_hasher).to_managed();
    }

    fn warn_missing(sess: Session, name: &str, default: &str) {
        if !*sess.building_library { return; }
        sess.warn(fmt!("missing crate link meta `%s`, using `%s` as default",
                       name, default));
    }

    fn crate_meta_name(sess: Session, output: &Path, opt_name: Option<@str>)
        -> @str {
        match opt_name {
            Some(v) => v,
            None => {
                // to_managed could go away if there was a version of
                // filestem that returned an @str
                let name = session::expect(sess,
                                           output.filestem(),
                                           || fmt!("output file name `%s` doesn't\
                                                    appear to have a stem",
                                                   output.to_str())).to_managed();
                warn_missing(sess, "name", name);
                name
            }
        }
    }

    fn crate_meta_vers(sess: Session, opt_vers: Option<@str>) -> @str {
        match opt_vers {
            Some(v) => v,
            None => {
                let vers = @"0.0";
                warn_missing(sess, "vers", vers);
                vers
            }
        }
    }

    let ProvidedMetas {
        name: opt_name,
        vers: opt_vers,
        cmh_items: cmh_items
    } = provided_link_metas(sess, c);
    let name = crate_meta_name(sess, output, opt_name);
    let vers = crate_meta_vers(sess, opt_vers);
    let dep_hashes = cstore::get_dep_hashes(sess.cstore);
    let extras_hash =
        crate_meta_extras_hash(symbol_hasher, cmh_items,
                               dep_hashes);

    LinkMeta {
        name: name,
        vers: vers,
        extras_hash: extras_hash
    }
}

pub fn truncated_hash_result(symbol_hasher: &mut hash::State) -> ~str {
    symbol_hasher.result_str()
}


// This calculates STH for a symbol, as defined above
pub fn symbol_hash(tcx: ty::ctxt,
                   symbol_hasher: &mut hash::State,
                   t: ty::t,
                   link_meta: LinkMeta)
                   -> @str {
    // NB: do *not* use abbrevs here as we want the symbol names
    // to be independent of one another in the crate.

    symbol_hasher.reset();
    write_string(symbol_hasher, link_meta.name);
    write_string(symbol_hasher, "-");
    write_string(symbol_hasher, link_meta.extras_hash);
    write_string(symbol_hasher, "-");
    write_string(symbol_hasher, encoder::encoded_ty(tcx, t));
    let mut hash = truncated_hash_result(symbol_hasher);
    // Prefix with _ so that it never blends into adjacent digits
    hash.unshift_char('_');
    // tjc: allocation is unfortunate; need to change core::hash
    hash.to_managed()
}

pub fn get_symbol_hash(ccx: &mut CrateContext, t: ty::t) -> @str {
    match ccx.type_hashcodes.find(&t) {
      Some(&h) => h,
      None => {
        let hash = symbol_hash(ccx.tcx, &mut ccx.symbol_hasher, t, ccx.link_meta);
        ccx.type_hashcodes.insert(t, hash);
        hash
      }
    }
}


// Name sanitation. LLVM will happily accept identifiers with weird names, but
// gas doesn't!
// gas accepts the following characters in symbols: a-z, A-Z, 0-9, ., _, $
pub fn sanitize(s: &str) -> ~str {
    let mut result = ~"";
    for s.iter().advance |c| {
        match c {
            // Escape these with $ sequences
            '@' => result += "$SP$",
            '~' => result += "$UP$",
            '*' => result += "$RP$",
            '&' => result += "$BP$",
            '<' => result += "$LT$",
            '>' => result += "$GT$",
            '(' => result += "$LP$",
            ')' => result += "$RP$",
            ',' => result += "$C$",

            // '.' doesn't occur in types and functions, so reuse it
            // for ':'
            ':' => result.push_char('.'),

            // These are legal symbols
            'a' .. 'z'
            | 'A' .. 'Z'
            | '0' .. '9'
            | '_' => result.push_char(c),

            _ => {
                if c > 'z' && char::is_XID_continue(c) {
                    result.push_char(c);
                }
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

pub fn mangle(sess: Session, ss: path) -> ~str {
    // Follow C++ namespace-mangling style

    let mut n = ~"_ZN"; // Begin name-sequence.

    for ss.iter().advance |s| {
        match *s { path_name(s) | path_mod(s) => {
          let sani = sanitize(sess.str_of(s));
          n += fmt!("%u%s", sani.len(), sani);
        } }
    }
    n += "E"; // End name-sequence.
    n
}

pub fn exported_name(sess: Session,
                     path: path,
                     hash: &str,
                     vers: &str) -> ~str {
    mangle(sess,
           vec::append_one(
               vec::append_one(path, path_name(sess.ident_of(hash))),
               path_name(sess.ident_of(vers))))
}

pub fn mangle_exported_name(ccx: &mut CrateContext,
                            path: path,
                            t: ty::t) -> ~str {
    let hash = get_symbol_hash(ccx, t);
    return exported_name(ccx.sess, path,
                         hash,
                         ccx.link_meta.vers);
}

pub fn mangle_internal_name_by_type_only(ccx: &mut CrateContext,
                                         t: ty::t,
                                         name: &str) -> ~str {
    let s = ppaux::ty_to_short_str(ccx.tcx, t);
    let hash = get_symbol_hash(ccx, t);
    return mangle(ccx.sess,
        ~[path_name(ccx.sess.ident_of(name)),
          path_name(ccx.sess.ident_of(s)),
          path_name(ccx.sess.ident_of(hash))]);
}

pub fn mangle_internal_name_by_type_and_seq(ccx: &mut CrateContext,
                                         t: ty::t,
                                         name: &str) -> ~str {
    let s = ppaux::ty_to_str(ccx.tcx, t);
    let hash = get_symbol_hash(ccx, t);
    return mangle(ccx.sess,
        ~[path_name(ccx.sess.ident_of(s)),
          path_name(ccx.sess.ident_of(hash)),
          path_name((ccx.names)(name))]);
}

pub fn mangle_internal_name_by_path_and_seq(ccx: &mut CrateContext,
                                            path: path,
                                            flav: &str) -> ~str {
    mangle(ccx.sess,
           vec::append_one(path, path_name((ccx.names)(flav))))
}

pub fn mangle_internal_name_by_path(ccx: &mut CrateContext, path: path) -> ~str {
    mangle(ccx.sess, path)
}

pub fn mangle_internal_name_by_seq(ccx: &mut CrateContext, flav: &str) -> ~str {
    fmt!("%s_%u", flav, (ccx.names)(flav).name)
}


pub fn output_dll_filename(os: session::os, lm: LinkMeta) -> ~str {
    let (dll_prefix, dll_suffix) = match os {
        session::os_win32 => (win32::DLL_PREFIX, win32::DLL_SUFFIX),
        session::os_macos => (macos::DLL_PREFIX, macos::DLL_SUFFIX),
        session::os_linux => (linux::DLL_PREFIX, linux::DLL_SUFFIX),
        session::os_android => (android::DLL_PREFIX, android::DLL_SUFFIX),
        session::os_freebsd => (freebsd::DLL_PREFIX, freebsd::DLL_SUFFIX),
    };
    fmt!("%s%s-%s-%s%s", dll_prefix, lm.name, lm.extras_hash, lm.vers, dll_suffix)
}

// If the user wants an exe generated we need to invoke
// cc to link the object file with some libs
pub fn link_binary(sess: Session,
                   obj_filename: &Path,
                   out_filename: &Path,
                   lm: LinkMeta) {
    // In the future, FreeBSD will use clang as default compiler.
    // It would be flexible to use cc (system's default C compiler)
    // instead of hard-coded gcc.
    // For win32, there is no cc command,
    // so we add a condition to make it use gcc.
    let cc_prog: ~str = match sess.opts.linker {
        Some(ref linker) => copy *linker,
        None => match sess.targ_cfg.os {
            session::os_android =>
                match &sess.opts.android_cross_path {
                    &Some(ref path) => {
                        fmt!("%s/bin/arm-linux-androideabi-gcc", *path)
                    }
                    &None => {
                        sess.fatal("need Android NDK path for linking \
                                    (--android-cross-path)")
                    }
                },
            session::os_win32 => ~"gcc",
            _ => ~"cc"
        }
    };
    // The invocations of cc share some flags across platforms


    let output = if *sess.building_library {
        let long_libname = output_dll_filename(sess.targ_cfg.os, lm);
        debug!("link_meta.name:  %s", lm.name);
        debug!("long_libname: %s", long_libname);
        debug!("out_filename: %s", out_filename.to_str());
        debug!("dirname(out_filename): %s", out_filename.dir_path().to_str());

        out_filename.dir_path().push(long_libname)
    } else {
        /*bad*/copy *out_filename
    };

    debug!("output: %s", output.to_str());
    let cc_args = link_args(sess, obj_filename, out_filename, lm);
    debug!("%s link args: %s", cc_prog, cc_args.connect(" "));
    // We run 'cc' here
    let prog = run::process_output(cc_prog, cc_args);
    if 0 != prog.status {
        sess.err(fmt!("linking with `%s` failed with code %d",
                      cc_prog, prog.status));
        sess.note(fmt!("%s arguments: %s",
                       cc_prog, cc_args.connect(" ")));
        sess.note(str::from_bytes(prog.error + prog.output));
        sess.abort_if_errors();
    }

    // Clean up on Darwin
    if sess.targ_cfg.os == session::os_macos {
        run::process_status("dsymutil", [output.to_str()]);
    }

    // Remove the temporary object file if we aren't saving temps
    if !sess.opts.save_temps {
        if ! os::remove_file(obj_filename) {
            sess.warn(fmt!("failed to delete object file `%s`",
                           obj_filename.to_str()));
        }
    }
}

pub fn link_args(sess: Session,
                 obj_filename: &Path,
                 out_filename: &Path,
                 lm:LinkMeta) -> ~[~str] {

    // Converts a library file-stem into a cc -l argument
    fn unlib(config: @session::config, stem: ~str) -> ~str {
        if stem.starts_with("lib") &&
            config.os != session::os_win32 {
            stem.slice(3, stem.len()).to_owned()
        } else {
            stem
        }
    }


    let output = if *sess.building_library {
        let long_libname = output_dll_filename(sess.targ_cfg.os, lm);
        out_filename.dir_path().push(long_libname)
    } else {
        /*bad*/copy *out_filename
    };

    // The default library location, we need this to find the runtime.
    // The location of crates will be determined as needed.
    let stage: ~str = ~"-L" + sess.filesearch.get_target_lib_path().to_str();

    let mut args = vec::append(~[stage], sess.targ_cfg.target_strs.cc_args);

    args.push_all([
        ~"-o", output.to_str(),
        obj_filename.to_str()]);

    let lib_cmd = match sess.targ_cfg.os {
        session::os_macos => ~"-dynamiclib",
        _ => ~"-shared"
    };

    // # Crate linking

    let cstore = sess.cstore;
    let r = cstore::get_used_crate_files(cstore);
    for r.iter().advance |cratepath| {
        if cratepath.filetype() == Some(~".rlib") {
            args.push(cratepath.to_str());
            loop;
        }
        let dir = cratepath.dirname();
        if dir != ~"" { args.push(~"-L" + dir); }
        let libarg = unlib(sess.targ_cfg, cratepath.filestem().get());
        args.push(~"-l" + libarg);
    }

    let ula = cstore::get_used_link_args(cstore);
    for ula.iter().advance |arg| { args.push(arg.to_owned()); }

    // Add all the link args for external crates.
    do cstore::iter_crate_data(cstore) |crate_num, _| {
        let link_args = csearch::get_link_args_for_crate(cstore, crate_num);
        do vec::consume(link_args) |_, link_arg| {
            args.push(link_arg);
        }
    }

    // # Extern library linking

    // User-supplied library search paths (-L on the cammand line) These are
    // the same paths used to find Rust crates, so some of them may have been
    // added already by the previous crate linking code. This only allows them
    // to be found at compile time so it is still entirely up to outside
    // forces to make sure that library can be found at runtime.

    for sess.opts.addl_lib_search_paths.iter().advance |path| {
        args.push(~"-L" + path.to_str());
    }

    // The names of the extern libraries
    let used_libs = cstore::get_used_libraries(cstore);
    for used_libs.iter().advance |l| { args.push(~"-l" + *l); }

    if *sess.building_library {
        args.push(lib_cmd);

        // On mac we need to tell the linker to let this library
        // be rpathed
        if sess.targ_cfg.os == session::os_macos {
            args.push(~"-Wl,-install_name,@rpath/"
                      + output.filename().get());
        }
    }

    // On linux librt and libdl are an indirect dependencies via rustrt,
    // and binutils 2.22+ won't add them automatically
    if sess.targ_cfg.os == session::os_linux {
        args.push_all([~"-lrt", ~"-ldl"]);

        // LLVM implements the `frem` instruction as a call to `fmod`,
        // which lives in libm. Similar to above, on some linuxes we
        // have to be explicit about linking to it. See #2510
        args.push(~"-lm");
    }
    else if sess.targ_cfg.os == session::os_android {
        args.push_all([~"-ldl", ~"-llog",  ~"-lsupc++", ~"-lgnustl_shared"]);
        args.push(~"-lm");
    }

    if sess.targ_cfg.os == session::os_freebsd {
        args.push_all([~"-pthread", ~"-lrt",
                       ~"-L/usr/local/lib", ~"-lexecinfo",
                       ~"-L/usr/local/lib/gcc46",
                       ~"-L/usr/local/lib/gcc44", ~"-lstdc++",
                       ~"-Wl,-z,origin",
                       ~"-Wl,-rpath,/usr/local/lib/gcc46",
                       ~"-Wl,-rpath,/usr/local/lib/gcc44"]);
    }

    // OS X 10.6 introduced 'compact unwind info', which is produced by the
    // linker from the dwarf unwind info. Unfortunately, it does not seem to
    // understand how to unwind our __morestack frame, so we have to turn it
    // off. This has impacted some other projects like GHC.
    if sess.targ_cfg.os == session::os_macos {
        args.push(~"-Wl,-no_compact_unwind");
    }

    // Stack growth requires statically linking a __morestack function
    args.push(~"-lmorestack");

    // Always want the runtime linked in
    args.push(~"-lrustrt");

    // FIXME (#2397): At some point we want to rpath our guesses as to where
    // extern libraries might live, based on the addl_lib_search_paths
    args.push_all(rpath::get_rpath_flags(sess, &output));

    // Finally add all the linker arguments provided on the command line
    args.push_all(sess.opts.linker_args);

    return args;
}
