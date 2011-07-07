
import driver::session;
import lib::llvm::llvm;
import front::attr;
import middle::trans;
import middle::ty;
import metadata::encoder;
import std::str;
import std::fs;
import std::vec;
import std::option;
import option::some;
import option::none;
import std::sha1::sha1;
import std::sort;
import trans::crate_ctxt;
import syntax::ast;
import syntax::print::pprust;
import lib::llvm::llvm::ModuleRef;
import lib::llvm::llvm::ValueRef;
import lib::llvm::mk_pass_manager;
import lib::llvm::mk_target_data;
import lib::llvm::mk_type_names;
import lib::llvm::False;
import lib::llvm::True;

tag output_type {
    output_type_none;
    output_type_bitcode;
    output_type_assembly;
    output_type_object;
    output_type_exe;
}

fn llvm_err(session::session sess, str msg) {
    auto buf = llvm::LLVMRustGetLastError();
    if (buf as uint == 0u) {
        sess.fatal(msg);
    } else { sess.fatal(msg + ": " + str::str_from_cstr(buf)); }
    fail;
}

fn link_intrinsics(session::session sess, ModuleRef llmod) {
    auto path = fs::connect(sess.get_opts().sysroot, "intrinsics.bc");
    auto membuf =
        llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(str::buf(path));
    if (membuf as uint == 0u) {
        llvm_err(sess, "installation problem: couldn't open " + path);
        fail;
    }
    auto llintrinsicsmod = llvm::LLVMRustParseBitcode(membuf);
    llvm::LLVMDisposeMemoryBuffer(membuf);
    if (llintrinsicsmod as uint == 0u) {
        llvm_err(sess, "installation problem: couldn't parse intrinsics.bc");
        fail;
    }
    auto linkres = llvm::LLVMLinkModules(llmod, llintrinsicsmod);
    llvm::LLVMDisposeModule(llintrinsicsmod);
    if (linkres == False) {
        llvm_err(sess, "couldn't link the module with the intrinsics");
        fail;
    }
}

mod write {
    fn is_object_or_assembly_or_exe(output_type ot) -> bool {
        if (ot == output_type_assembly || ot == output_type_object ||
                ot == output_type_exe) {
            ret true;
        }
        ret false;
    }

    // Decides what to call an intermediate file, given the name of the output
    // and the extension to use.
    fn mk_intermediate_name(str output_path, str extension) -> str {
        auto dot_pos = str::index(output_path, '.' as u8);
        auto stem;
        if (dot_pos < 0) {
            stem = output_path;
        } else { stem = str::substr(output_path, 0u, dot_pos as uint); }
        ret stem + "." + extension;
    }
    fn run_passes(session::session sess, ModuleRef llmod, str output) {
        auto opts = sess.get_opts();
        if (opts.time_llvm_passes) { llvm::LLVMRustEnableTimePasses(); }
        link_intrinsics(sess, llmod);
        auto pm = mk_pass_manager();
        auto td = mk_target_data(x86::get_data_layout());
        llvm::LLVMAddTargetData(td.lltd, pm.llpm);
        // TODO: run the linter here also, once there are llvm-c bindings for
        // it.

        // Generate a pre-optimization intermediate file if -save-temps was
        // specified.

        if (opts.save_temps) {
            alt (opts.output_type) {
                case (output_type_bitcode) {
                    if (opts.optimize != 0u) {
                        auto filename =
                            mk_intermediate_name(output, "no-opt.bc");
                        llvm::LLVMWriteBitcodeToFile(llmod,
                                                     str::buf(filename));
                    }
                }
                case (_) {
                    auto filename = mk_intermediate_name(output, "bc");
                    llvm::LLVMWriteBitcodeToFile(llmod, str::buf(filename));
                }
            }
        }
        if (opts.verify) { llvm::LLVMAddVerifierPass(pm.llpm); }
        // FIXME: This is mostly a copy of the bits of opt's -O2 that are
        // available in the C api.
        // FIXME2: We might want to add optimization levels like -O1, -O2,
        // -Os, etc
        // FIXME3: Should we expose and use the pass lists used by the opt
        // tool?

        if (opts.optimize != 0u) {
            auto fpm = mk_pass_manager();
            llvm::LLVMAddTargetData(td.lltd, fpm.llpm);
            llvm::LLVMAddStandardFunctionPasses(fpm.llpm, 2u);
            llvm::LLVMRunPassManager(fpm.llpm, llmod);
            let uint threshold = 225u;
            if (opts.optimize == 3u) { threshold = 275u; }
            llvm::LLVMAddStandardModulePasses(pm.llpm,
                                              // optimization level
                                              opts.optimize,
                                              False, // optimize for size
                                               True, // unit-at-a-time
                                               True, // unroll loops
                                               True, // simplify lib calls
                                               threshold); // inline threshold

        }
        if (opts.verify) { llvm::LLVMAddVerifierPass(pm.llpm); }
        if (is_object_or_assembly_or_exe(opts.output_type)) {
            let int LLVMAssemblyFile = 0;
            let int LLVMObjectFile = 1;
            let int LLVMOptNone = 0;        // -O0
            let int LLVMOptLess = 1;        // -O1
            let int LLVMOptDefault = 2;     // -O2, -Os
            let int LLVMOptAggressive = 3;  // -O3

            auto CodeGenOptLevel;
            alt (opts.optimize) {
                case (0u) {
                     CodeGenOptLevel = LLVMOptNone;
                }
                case (1u) {
                     CodeGenOptLevel = LLVMOptLess;
                }
                case (2u) {
                     CodeGenOptLevel = LLVMOptDefault;
                }
                case (3u) {
                     CodeGenOptLevel = LLVMOptAggressive;
                }
                case (_) {
                     fail;
                }
            }

            auto FileType;
            if (opts.output_type == output_type_object ||
                    opts.output_type == output_type_exe) {
                FileType = LLVMObjectFile;
            } else { FileType = LLVMAssemblyFile; }
            // Write optimized bitcode if --save-temps was on.

            if (opts.save_temps) {
                // Always output the bitcode file with --save-temps

                auto filename = mk_intermediate_name(output, "opt.bc");
                llvm::LLVMRunPassManager(pm.llpm, llmod);
                llvm::LLVMWriteBitcodeToFile(llmod, str::buf(filename));
                pm = mk_pass_manager();
                // Save the assembly file if -S is used

                if (opts.output_type == output_type_assembly) {
                    auto triple = x86::get_target_triple();
                    llvm::LLVMRustWriteOutputFile(pm.llpm, llmod,
                                                  str::buf(triple),
                                                  str::buf(output),
                                                  LLVMAssemblyFile,
                                                  CodeGenOptLevel);
                }

                // Save the object file for -c or --save-temps alone
                // This .o is needed when an exe is built
                if (opts.output_type == output_type_object ||
                        opts.output_type == output_type_exe) {
                    auto triple = x86::get_target_triple();
                    llvm::LLVMRustWriteOutputFile(pm.llpm, llmod,
                                                  str::buf(triple),
                                                  str::buf(output),
                                                  LLVMObjectFile,
                                                  CodeGenOptLevel);
                }
            } else {
                // If we aren't saving temps then just output the file
                // type corresponding to the '-c' or '-S' flag used

                auto triple = x86::get_target_triple();
                llvm::LLVMRustWriteOutputFile(pm.llpm, llmod,
                                              str::buf(triple),
                                              str::buf(output), FileType,
                                              CodeGenOptLevel);
            }
            // Clean up and return

            llvm::LLVMDisposeModule(llmod);
            if (opts.time_llvm_passes) { llvm::LLVMRustPrintPassTimings(); }
            ret;
        }
        // If only a bitcode file is asked for by using the '--emit-llvm'
        // flag, then output it here

        llvm::LLVMRunPassManager(pm.llpm, llmod);
        llvm::LLVMWriteBitcodeToFile(llmod, str::buf(output));
        llvm::LLVMDisposeModule(llmod);
        if (opts.time_llvm_passes) { llvm::LLVMRustPrintPassTimings(); }
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
 *  - Linnkers operate on a flat namespace, so we have to flatten names.
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
 *  - Define CMH as hash(CMETA).
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

type link_meta = rec(str name,
                     str vers,
                     str extras_hash);

fn build_link_meta(&session::session sess, &ast::crate c,
                   &str output, sha1 sha) -> link_meta {

    type provided_metas = rec(option::t[str] name,
                              option::t[str] vers,
                              vec[@ast::meta_item] cmh_items);

    fn provided_link_metas(&session::session sess,
                           &ast::crate c) -> provided_metas {
        let option::t[str] name = none;
        let option::t[str] vers = none;
        let vec[@ast::meta_item] cmh_items = [];
        auto linkage_metas = attr::find_linkage_metas(c.node.attrs);
        attr::require_unique_names(sess, linkage_metas);
        for (@ast::meta_item meta in linkage_metas) {
            if (attr::get_meta_item_name(meta) == "name") {
                alt (attr::get_meta_item_value_str(meta)) {
                    case (some(?v)) { name = some(v); }
                    case (none) { cmh_items += [meta]; }
                }
            } else if (attr::get_meta_item_name(meta) == "vers") {
                alt (attr::get_meta_item_value_str(meta)) {
                    case (some(?v)) { vers = some(v); }
                    case (none) { cmh_items += [meta]; }
                }
            } else {
                cmh_items += [meta];
            }
        }
        ret rec(name = name,
                vers = vers,
                cmh_items = cmh_items);
    }

    // This calculates CMH as defined above
    fn crate_meta_extras_hash(sha1 sha, &ast::crate crate,
                              &provided_metas metas) -> str {
        fn len_and_str(&str s) -> str {
            ret #fmt("%u_%s", str::byte_len(s), s);
        }

        fn len_and_str_lit(&ast::lit l) -> str {
            ret len_and_str(pprust::lit_to_str(@l));
        }
    
        auto cmh_items = attr::sort_meta_items(metas.cmh_items);

        sha.reset();
        for (@ast::meta_item m_ in cmh_items) {
            auto m = m_;
            alt (m.node) {
                case (ast::meta_name_value(?key, ?value)) {
                    sha.input_str(len_and_str(key));
                    sha.input_str(len_and_str_lit(value));
                }
                case (ast::meta_word(?name)) {
                    sha.input_str(len_and_str(name));
                }
                case (ast::meta_list(_, _)) {
                    // FIXME (#607): Implement this
                    fail "unimplemented meta_item variant";
                }
            }
        }
        ret truncated_sha1_result(sha);
    }

    fn warn_missing(&session::session sess, str name, str default) {
        if (!sess.get_opts().library) { ret; }
        sess.warn(#fmt("missing crate link meta '%s', using '%s' as default",
                       name, default));
    }

    fn crate_meta_name(&session::session sess, &ast::crate crate,
                       &str output, &provided_metas metas) -> str {
        ret alt (metas.name) {
            case (some(?v)) { v }
            case (none) {
                auto name = {
                    auto os = str::split(fs::basename(output), '.' as u8);
                    assert (vec::len(os) >= 2u);
                    vec::pop(os);
                    str::connect(os, ".")
                };
                warn_missing(sess, "name", name);
                name
            }
        };
    }

    fn crate_meta_vers(&session::session sess, &ast::crate crate,
                       &provided_metas metas) -> str {
        ret alt (metas.vers) {
            case (some(?v)) { v }
            case (none) {
                auto vers = "0.0";
                warn_missing(sess, "vers", vers);
                vers
            }
        };
    }

    auto provided_metas = provided_link_metas(sess, c);
    auto name = crate_meta_name(sess, c, output, provided_metas);
    auto vers = crate_meta_vers(sess, c, provided_metas);
    auto extras_hash = crate_meta_extras_hash(sha, c, provided_metas);

    ret rec(name = name, vers = vers, extras_hash = extras_hash);
}

fn truncated_sha1_result(sha1 sha) -> str {
    ret str::substr(sha.result_str(), 0u, 16u);
}


// This calculates STH for a symbol, as defined above
fn symbol_hash(ty::ctxt tcx, sha1 sha, &ty::t t,
               &link_meta link_meta) -> str {
    // NB: do *not* use abbrevs here as we want the symbol names
    // to be independent of one another in the crate.

    sha.reset();
    sha.input_str(link_meta.name);
    sha.input_str("-");
    // FIXME: This wants to be link_meta.meta_hash
    sha.input_str(link_meta.name);
    sha.input_str("-");
    sha.input_str(encoder::encoded_ty(tcx, t));
    auto hash = truncated_sha1_result(sha);
    // Prefix with _ so that it never blends into adjacent digits

    ret "_" + hash;
}

fn get_symbol_hash(&@crate_ctxt ccx, &ty::t t) -> str {
    auto hash = "";
    alt (ccx.type_sha1s.find(t)) {
        case (some(?h)) { hash = h; }
        case (none) {
            hash =
                symbol_hash(ccx.tcx, ccx.sha, t,
                            ccx.link_meta);
            ccx.type_sha1s.insert(t, hash);
        }
    }
    ret hash;
}

fn mangle(&vec[str] ss) -> str {
    // Follow C++ namespace-mangling style

    auto n = "_ZN"; // Begin name-sequence.

    for (str s in ss) { n += #fmt("%u%s", str::byte_len(s), s); }
    n += "E"; // End name-sequence.

    ret n;
}

fn exported_name(&vec[str] path, &str hash, &str vers) -> str {
    // FIXME: versioning isn't working yet

    ret mangle(path + [hash]); //  + "@" + vers;

}

fn mangle_exported_name(&@crate_ctxt ccx, &vec[str] path, &ty::t t) -> str {
    auto hash = get_symbol_hash(ccx, t);
    ret exported_name(path, hash, ccx.link_meta.vers);
}

fn mangle_internal_name_by_type_only(&@crate_ctxt ccx, &ty::t t, &str name) ->
   str {
    auto s = util::ppaux::ty_to_short_str(ccx.tcx, t);
    auto hash = get_symbol_hash(ccx, t);
    ret mangle([name, s, hash]);
}

fn mangle_internal_name_by_path_and_seq(&@crate_ctxt ccx, &vec[str] path,
                                        &str flav) -> str {
    ret mangle(path + [ccx.names.next(flav)]);
}

fn mangle_internal_name_by_path(&@crate_ctxt ccx, &vec[str] path) -> str {
    ret mangle(path);
}

fn mangle_internal_name_by_seq(&@crate_ctxt ccx, &str flav) -> str {
    ret ccx.names.next(flav);
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
