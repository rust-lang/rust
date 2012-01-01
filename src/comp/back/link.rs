
import driver::session;
import lib::llvm::llvm;
import front::attr;
import middle::ty;
import metadata::{encoder, cstore};
import middle::trans_common::crate_ctxt;
import str;
import std::fs;
import vec;
import option;
import std::run;
import option::some;
import option::none;
import std::sha1::sha1;
import syntax::ast;
import syntax::print::pprust;
import lib::llvm::llvm::ModuleRef;
import lib::llvm::mk_pass_manager;
import lib::llvm::mk_target_data;
import lib::llvm::False;
import util::filesearch;

tag output_type {
    output_type_none;
    output_type_bitcode;
    output_type_assembly;
    output_type_llvm_assembly;
    output_type_object;
    output_type_exe;
}

fn llvm_err(sess: session::session, msg: str) unsafe {
    let buf = llvm::LLVMRustGetLastError();
    if buf == ptr::null() {
        sess.fatal(msg);
    } else { sess.fatal(msg + ": " + str::str_from_cstr(buf)); }
}

fn load_intrinsics_bc(sess: session::session) -> option::t<ModuleRef> {
    let path = alt filesearch::search(
        sess.filesearch(),
        bind filesearch::pick_file("intrinsics.bc", _)) {
      option::some(path) { path }
      option::none. {
        sess.warn("couldn't find intrinsics.bc");
        ret option::none;
      }
    };
    let membuf = str::as_buf(path, {|buf|
        llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(buf)
                                   });
    if membuf as uint == 0u {
        llvm_err(sess, "installation problem: couldn't open " + path);
        fail;
    }
    let llintrinsicsmod = llvm::LLVMRustParseBitcode(membuf);
    llvm::LLVMDisposeMemoryBuffer(membuf);
    if llintrinsicsmod as uint == 0u {
        sess.warn("couldn't parse intrinsics.bc");
        ret option::none;
    }

    ret option::some(llintrinsicsmod);
}

fn load_intrinsics_ll(sess: session::session) -> ModuleRef {
    let path = alt filesearch::search(
        sess.filesearch(),
        bind filesearch::pick_file("intrinsics.ll", _)) {
      option::some(path) { path }
      option::none. { sess.fatal("couldn't find intrinsics.ll") }
    };
    let llintrinsicsmod = str::as_buf(path, { |buf|
        llvm::LLVMRustParseAssemblyFile(buf)
                                        });
    if llintrinsicsmod as uint == 0u {
        llvm_err(sess, "couldn't parse intrinsics.ll");
        fail;
    }
    ret llintrinsicsmod;
}

fn link_intrinsics(sess: session::session, llmod: ModuleRef) {
    let llintrinsicsmod = {
        alt load_intrinsics_bc(sess) {
          option::some(m) { m }
          option::none. {
            // When the bitcode format changes we can't parse a .bc
            // file produced with a newer LLVM (as happens when stage0
            // is trying to build against a new LLVM revision), in
            // that case we'll try to parse the assembly.
            sess.warn("couldn't parse intrinsics.bc, trying intrinsics.ll");
            load_intrinsics_ll(sess)
          }
        }
    };
    let linkres = llvm::LLVMLinkModules(llmod, llintrinsicsmod);
    llvm::LLVMDisposeModule(llintrinsicsmod);
    if linkres == False {
        llvm_err(sess, "couldn't link the module with the intrinsics");
        fail;
    }
}

mod write {
    fn is_object_or_assembly_or_exe(ot: output_type) -> bool {
        if ot == output_type_assembly || ot == output_type_object ||
               ot == output_type_exe {
            ret true;
        }
        ret false;
    }

    // Decides what to call an intermediate file, given the name of the output
    // and the extension to use.
    fn mk_intermediate_name(output_path: str, extension: str) -> str {
        let dot_pos = str::index(output_path, '.' as u8);
        let stem;
        if dot_pos < 0 {
            stem = output_path;
        } else { stem = str::substr(output_path, 0u, dot_pos as uint); }
        ret stem + "." + extension;
    }
    fn run_passes(sess: session::session, llmod: ModuleRef, output: str) {
        let opts = sess.get_opts();
        if opts.time_llvm_passes { llvm::LLVMRustEnableTimePasses(); }
        link_intrinsics(sess, llmod);
        let pm = mk_pass_manager();
        let td = mk_target_data(
            sess.get_targ_cfg().target_strs.data_layout);
        llvm::LLVMAddTargetData(td.lltd, pm.llpm);
        // TODO: run the linter here also, once there are llvm-c bindings for
        // it.

        // Generate a pre-optimization intermediate file if -save-temps was
        // specified.

        if opts.save_temps {
            alt opts.output_type {
              output_type_bitcode. {
                if opts.optimize != 0u {
                    let filename = mk_intermediate_name(output, "no-opt.bc");
                    str::as_buf(filename,
                                {|buf|
                                    llvm::LLVMWriteBitcodeToFile(llmod, buf)
                                });
                }
              }
              _ {
                let filename = mk_intermediate_name(output, "bc");
                str::as_buf(filename,
                            {|buf|
                                llvm::LLVMWriteBitcodeToFile(llmod, buf)
                            });
              }
            }
        }
        if opts.verify { llvm::LLVMAddVerifierPass(pm.llpm); }
        // FIXME: This is mostly a copy of the bits of opt's -O2 that are
        // available in the C api.
        // FIXME2: We might want to add optimization levels like -O1, -O2,
        // -Os, etc
        // FIXME3: Should we expose and use the pass lists used by the opt
        // tool?

        if opts.optimize != 0u {
            let fpm = mk_pass_manager();
            llvm::LLVMAddTargetData(td.lltd, fpm.llpm);

            let FPMB = llvm::LLVMPassManagerBuilderCreate();
            llvm::LLVMPassManagerBuilderSetOptLevel(FPMB, 2u);
            llvm::LLVMPassManagerBuilderPopulateFunctionPassManager(FPMB,
                                                                    fpm.llpm);
            llvm::LLVMPassManagerBuilderDispose(FPMB);

            llvm::LLVMRunPassManager(fpm.llpm, llmod);
            let threshold: uint = 225u;
            if opts.optimize == 3u { threshold = 275u; }

            let MPMB = llvm::LLVMPassManagerBuilderCreate();
            llvm::LLVMPassManagerBuilderSetOptLevel(MPMB, opts.optimize);
            llvm::LLVMPassManagerBuilderSetSizeLevel(MPMB, 0);
            llvm::LLVMPassManagerBuilderSetDisableUnitAtATime(MPMB, False);
            llvm::LLVMPassManagerBuilderSetDisableUnrollLoops(MPMB, False);
            llvm::LLVMPassManagerBuilderSetDisableSimplifyLibCalls(MPMB,
                                                                   False);

            if threshold != 0u {
                llvm::LLVMPassManagerBuilderUseInlinerWithThreshold
                    (MPMB, threshold);
            }
            llvm::LLVMPassManagerBuilderPopulateModulePassManager(MPMB,
                                                                  pm.llpm);

            llvm::LLVMPassManagerBuilderDispose(MPMB);
        }
        if opts.verify { llvm::LLVMAddVerifierPass(pm.llpm); }
        if is_object_or_assembly_or_exe(opts.output_type) {
            let LLVMAssemblyFile: int = 0;
            let LLVMObjectFile: int = 1;
            let LLVMOptNone: int = 0; // -O0
            let LLVMOptLess: int = 1; // -O1
            let LLVMOptDefault: int = 2; // -O2, -Os
            let LLVMOptAggressive: int = 3; // -O3

            let CodeGenOptLevel;
            alt opts.optimize {
              0u { CodeGenOptLevel = LLVMOptNone; }
              1u { CodeGenOptLevel = LLVMOptLess; }
              2u { CodeGenOptLevel = LLVMOptDefault; }
              3u { CodeGenOptLevel = LLVMOptAggressive; }
              _ { fail; }
            }

            let FileType;
            if opts.output_type == output_type_object ||
                   opts.output_type == output_type_exe {
                FileType = LLVMObjectFile;
            } else { FileType = LLVMAssemblyFile; }
            // Write optimized bitcode if --save-temps was on.

            if opts.save_temps {
                // Always output the bitcode file with --save-temps

                let filename = mk_intermediate_name(output, "opt.bc");
                llvm::LLVMRunPassManager(pm.llpm, llmod);
                str::as_buf(filename,
                            {|buf|
                                llvm::LLVMWriteBitcodeToFile(llmod, buf)
                            });
                pm = mk_pass_manager();
                // Save the assembly file if -S is used

                if opts.output_type == output_type_assembly {
                    let _: () = str::as_buf(
                        sess.get_targ_cfg().target_strs.target_triple,
                        {|buf_t|
                            str::as_buf(output, {|buf_o|
                                llvm::LLVMRustWriteOutputFile(
                                    pm.llpm,
                                    llmod,
                                    buf_t,
                                    buf_o,
                                    LLVMAssemblyFile,
                                    CodeGenOptLevel,
                                    true)})});
                }


                // Save the object file for -c or --save-temps alone
                // This .o is needed when an exe is built
                if opts.output_type == output_type_object ||
                       opts.output_type == output_type_exe {
                    let _: () =
                        str::as_buf(
                            sess.get_targ_cfg().target_strs.target_triple,
                            {|buf_t|
                                str::as_buf(output, {|buf_o|
                                    llvm::LLVMRustWriteOutputFile(
                                        pm.llpm,
                                        llmod,
                                        buf_t,
                                        buf_o,
                                        LLVMObjectFile,
                                        CodeGenOptLevel,
                                        true)})});
                }
            } else {
                // If we aren't saving temps then just output the file
                // type corresponding to the '-c' or '-S' flag used

                let _: () =
                    str::as_buf(
                        sess.get_targ_cfg().target_strs.target_triple,
                        {|buf_t|
                            str::as_buf(output, {|buf_o|
                                llvm::LLVMRustWriteOutputFile(
                                    pm.llpm,
                                    llmod,
                                    buf_t,
                                    buf_o,
                                    FileType,
                                    CodeGenOptLevel,
                                    true)})});
            }
            // Clean up and return

            llvm::LLVMDisposeModule(llmod);
            if opts.time_llvm_passes { llvm::LLVMRustPrintPassTimings(); }
            ret;
        }

        if opts.output_type == output_type_llvm_assembly {
            // Given options "-S --emit-llvm": output LLVM assembly
            str::as_buf(output, {|buf_o|
                llvm::LLVMRustAddPrintModulePass(pm.llpm, llmod, buf_o)});
        } else {
            // If only a bitcode file is asked for by using the '--emit-llvm'
            // flag, then output it here
            llvm::LLVMRunPassManager(pm.llpm, llmod);
            str::as_buf(output,
                        {|buf| llvm::LLVMWriteBitcodeToFile(llmod, buf) });
        }

        llvm::LLVMDisposeModule(llmod);
        if opts.time_llvm_passes { llvm::LLVMRustPrintPassTimings(); }
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

type link_meta = {name: str, vers: str, extras_hash: str};

fn build_link_meta(sess: session::session, c: ast::crate, output: str,
                   sha: sha1) -> link_meta {

    type provided_metas =
        {name: option::t<str>,
         vers: option::t<str>,
         cmh_items: [@ast::meta_item]};

    fn provided_link_metas(sess: session::session, c: ast::crate) ->
       provided_metas {
        let name: option::t<str> = none;
        let vers: option::t<str> = none;
        let cmh_items: [@ast::meta_item] = [];
        let linkage_metas = attr::find_linkage_metas(c.node.attrs);
        attr::require_unique_names(sess, linkage_metas);
        for meta: @ast::meta_item in linkage_metas {
            if attr::get_meta_item_name(meta) == "name" {
                alt attr::get_meta_item_value_str(meta) {
                  some(v) { name = some(v); }
                  none. { cmh_items += [meta]; }
                }
            } else if attr::get_meta_item_name(meta) == "vers" {
                alt attr::get_meta_item_value_str(meta) {
                  some(v) { vers = some(v); }
                  none. { cmh_items += [meta]; }
                }
            } else { cmh_items += [meta]; }
        }
        ret {name: name, vers: vers, cmh_items: cmh_items};
    }

    // This calculates CMH as defined above
    fn crate_meta_extras_hash(sha: sha1, _crate: ast::crate,
                              metas: provided_metas,
                              dep_hashes: [str]) -> str {
        fn len_and_str(s: str) -> str {
            ret #fmt["%u_%s", str::byte_len(s), s];
        }

        fn len_and_str_lit(l: ast::lit) -> str {
            ret len_and_str(pprust::lit_to_str(@l));
        }

        let cmh_items = attr::sort_meta_items(metas.cmh_items);

        sha.reset();
        for m_: @ast::meta_item in cmh_items {
            let m = m_;
            alt m.node {
              ast::meta_name_value(key, value) {
                sha.input_str(len_and_str(key));
                sha.input_str(len_and_str_lit(value));
              }
              ast::meta_word(name) { sha.input_str(len_and_str(name)); }
              ast::meta_list(_, _) {
                // FIXME (#607): Implement this
                fail "unimplemented meta_item variant";
              }
            }
        }

        for dh in dep_hashes {
            sha.input_str(len_and_str(dh));
        }

        ret truncated_sha1_result(sha);
    }

    fn warn_missing(sess: session::session, name: str, default: str) {
        if !sess.building_library() { ret; }
        sess.warn(#fmt["missing crate link meta '%s', using '%s' as default",
                       name, default]);
    }

    fn crate_meta_name(sess: session::session, _crate: ast::crate,
                       output: str, metas: provided_metas) -> str {
        ret alt metas.name {
              some(v) { v }
              none. {
                let name =
                    {
                        let os = str::split(fs::basename(output), '.' as u8);
                        assert (vec::len(os) >= 2u);
                        vec::pop(os);
                        str::connect(os, ".")
                    };
                warn_missing(sess, "name", name);
                name
              }
            };
    }

    fn crate_meta_vers(sess: session::session, _crate: ast::crate,
                       metas: provided_metas) -> str {
        ret alt metas.vers {
              some(v) { v }
              none. {
                let vers = "0.0";
                warn_missing(sess, "vers", vers);
                vers
              }
            };
    }

    let provided_metas = provided_link_metas(sess, c);
    let name = crate_meta_name(sess, c, output, provided_metas);
    let vers = crate_meta_vers(sess, c, provided_metas);
    let dep_hashes = cstore::get_dep_hashes(sess.get_cstore());
    let extras_hash =
        crate_meta_extras_hash(sha, c, provided_metas, dep_hashes);

    ret {name: name, vers: vers, extras_hash: extras_hash};
}

fn truncated_sha1_result(sha: sha1) -> str {
    ret str::substr(sha.result_str(), 0u, 16u);
}


// This calculates STH for a symbol, as defined above
fn symbol_hash(tcx: ty::ctxt, sha: sha1, t: ty::t, link_meta: link_meta) ->
   str {
    // NB: do *not* use abbrevs here as we want the symbol names
    // to be independent of one another in the crate.

    sha.reset();
    sha.input_str(link_meta.name);
    sha.input_str("-");
    // FIXME: This wants to be link_meta.meta_hash
    sha.input_str(link_meta.name);
    sha.input_str("-");
    sha.input_str(encoder::encoded_ty(tcx, t));
    let hash = truncated_sha1_result(sha);
    // Prefix with _ so that it never blends into adjacent digits

    ret "_" + hash;
}

fn get_symbol_hash(ccx: @crate_ctxt, t: ty::t) -> str {
    let hash = "";
    alt ccx.type_sha1s.find(t) {
      some(h) { hash = h; }
      none. {
        hash = symbol_hash(ccx.tcx, ccx.sha, t, ccx.link_meta);
        ccx.type_sha1s.insert(t, hash);
      }
    }
    ret hash;
}

fn mangle(ss: [str]) -> str {
    // Follow C++ namespace-mangling style

    let n = "_ZN"; // Begin name-sequence.

    for s: str in ss { n += #fmt["%u%s", str::byte_len(s), s]; }
    n += "E"; // End name-sequence.

    ret n;
}

fn exported_name(path: [str], hash: str, _vers: str) -> str {
    // FIXME: versioning isn't working yet

    ret mangle(path + [hash]); //  + "@" + vers;

}

fn mangle_exported_name(ccx: @crate_ctxt, path: [str], t: ty::t) -> str {
    let hash = get_symbol_hash(ccx, t);
    ret exported_name(path, hash, ccx.link_meta.vers);
}

fn mangle_internal_name_by_type_only(ccx: @crate_ctxt, t: ty::t, name: str) ->
   str {
    let s = util::ppaux::ty_to_short_str(ccx.tcx, t);
    let hash = get_symbol_hash(ccx, t);
    ret mangle([name, s, hash]);
}

fn mangle_internal_name_by_path_and_seq(ccx: @crate_ctxt, path: [str],
                                        flav: str) -> str {
    ret mangle(path + [ccx.names.next(flav)]);
}

fn mangle_internal_name_by_path(_ccx: @crate_ctxt, path: [str]) -> str {
    ret mangle(path);
}

fn mangle_internal_name_by_seq(ccx: @crate_ctxt, flav: str) -> str {
    ret ccx.names.next(flav);
}

// If the user wants an exe generated we need to invoke
// gcc to link the object file with some libs
fn link_binary(sess: session::session,
               obj_filename: str,
               out_filename: str,
               lm: link_meta) {
    // Converts a library file name into a gcc -l argument
    fn unlib(config: @session::config, filename: str) -> str {
        let rmlib =
            bind fn (config: @session::config, filename: str) -> str {
                     if config.os == session::os_macos ||
                            (config.os == session::os_linux ||
                             config.os == session::os_freebsd) &&
                                str::find(filename, "lib") == 0 {
                         ret str::slice(filename, 3u,
                                        str::byte_len(filename));
                     } else { ret filename; }
                 }(config, _);
        fn rmext(filename: str) -> str {
            let parts = str::split(filename, '.' as u8);
            vec::pop(parts);
            ret str::connect(parts, ".");
        }
        ret alt config.os {
              session::os_macos. { rmext(rmlib(filename)) }
              session::os_linux. { rmext(rmlib(filename)) }
              session::os_freebsd. { rmext(rmlib(filename)) }
              _ { rmext(filename) }
            };
    }

    let output = if sess.building_library() {
        let long_libname =
            std::os::dylib_filename(#fmt("%s-%s-%s",
                                         lm.name, lm.extras_hash, lm.vers));
        #debug("link_meta.name:  %s", lm.name);
        #debug("long_libname: %s", long_libname);
        #debug("out_filename: %s", out_filename);
        #debug("dirname(out_filename): %s", fs::dirname(out_filename));

        fs::connect(fs::dirname(out_filename), long_libname)
    } else { out_filename };

    log(debug, "output: " + output);

    // The default library location, we need this to find the runtime.
    // The location of crates will be determined as needed.
    let stage: str = "-L" + sess.filesearch().get_target_lib_path();

    let prog: str = "gcc";
    // The invocations of gcc share some flags across platforms

    let gcc_args =
        [stage] + sess.get_targ_cfg().target_strs.gcc_args +
        ["-o", output, obj_filename];

    let lib_cmd;
    let os = sess.get_targ_cfg().os;
    if os == session::os_macos {
        lib_cmd = "-dynamiclib";
    } else { lib_cmd = "-shared"; }

    let cstore = sess.get_cstore();
    for cratepath: str in cstore::get_used_crate_files(cstore) {
        if str::ends_with(cratepath, ".rlib") {
            gcc_args += [cratepath];
            cont;
        }
        let cratepath = cratepath;
        let dir = fs::dirname(cratepath);
        if dir != "" { gcc_args += ["-L" + dir]; }
        let libarg = unlib(sess.get_targ_cfg(), fs::basename(cratepath));
        gcc_args += ["-l" + libarg];
    }

    let ula = cstore::get_used_link_args(cstore);
    for arg: str in ula { gcc_args += [arg]; }

    let used_libs = cstore::get_used_libraries(cstore);
    for l: str in used_libs { gcc_args += ["-l" + l]; }

    if sess.building_library() {
        gcc_args += [lib_cmd];

        // On mac we need to tell the linker to let this library
        // be rpathed
        if sess.get_targ_cfg().os == session::os_macos {
            gcc_args += ["-Wl,-install_name,@rpath/"
                        + fs::basename(output)];
        }
    } else {
        // FIXME: why do we hardcode -lm?
        gcc_args += ["-lm"];
    }

    // Always want the runtime linked in
    gcc_args += ["-lrustrt"];

    // On linux librt and libdl are an indirect dependencies via rustrt,
    // and binutils 2.22+ won't add them automatically
    if sess.get_targ_cfg().os == session::os_linux {
        gcc_args += ["-lrt", "-ldl"];
    }

    if sess.get_targ_cfg().os == session::os_freebsd {
        gcc_args += ["-lrt", "-L/usr/local/lib", "-lexecinfo",
                     "-L/usr/local/lib/gcc46",
                     "-L/usr/local/lib/gcc44", "-lstdc++",
                     "-Wl,-z,origin"];
    }

    // OS X 10.6 introduced 'compact unwind info', which is produced by the
    // linker from the dwarf unwind info. Unfortunately, it does not seem to
    // understand how to unwind our __morestack frame, so we have to turn it
    // off. This has impacted some other projects like GHC.
    if sess.get_targ_cfg().os == session::os_macos {
        gcc_args += ["-Wl,-no_compact_unwind"];
    }

    // Stack growth requires statically linking a __morestack function
    gcc_args += ["-lmorestack"];

    gcc_args += rpath::get_rpath_flags(sess, output);

    #debug("gcc link args: %s", str::connect(gcc_args, " "));
    // We run 'gcc' here
    let prog = run::program_output(prog, gcc_args);
    if 0 != prog.status {
        sess.err(#fmt["linking with gcc failed with code %d", prog.status]);
        sess.note(#fmt["gcc arguments: %s", str::connect(gcc_args, " ")]);
        sess.note(prog.err + prog.out);
        sess.abort_if_errors();
    }

    // Clean up on Darwin
    if sess.get_targ_cfg().os == session::os_macos {
        run::run_program("dsymutil", [output]);
    }

    // Remove the temporary object file if we aren't saving temps
    if !sess.get_opts().save_temps {
        run::run_program("rm", [obj_filename]);
    }
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
