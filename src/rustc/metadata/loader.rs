#[doc = "

Finds crate binaries and loads their metadata

"];

import driver::session;
import session::session;
import syntax::{ast, attr};
import syntax::print::pprust;
import syntax::codemap::span;
import lib::llvm::{False, llvm, mk_object_file, mk_section_iter};
import util::{filesearch};
import io::writer_util;

export load_library_crate;
export list_file_metadata;
export note_linkage_attrs;
export crate_name_from_metas;
export metadata_matches;

fn load_library_crate(sess: session::session, ident: ast::ident, span: span,
                      metas: [@ast::meta_item], hash: str)
   -> {ident: str, data: @[u8]} {


    alt find_library_crate(sess, span, metas, hash) {
      some(t) { ret t; }
      none {
        sess.span_fatal(span, #fmt["can't find crate for '%s'", ident]);
      }
    }
}

fn find_library_crate(sess: session::session, span: span,
                      metas: [@ast::meta_item], hash: str)
   -> option<{ident: str, data: @[u8]}> {

    attr::require_unique_names(sess.diagnostic(), metas);
    let metas = metas;

    let nn = default_native_lib_naming(sess, sess.opts.static);
    let x =
        find_library_crate_aux(sess, span, nn,
                               metas, hash, sess.filesearch);
    if x != none || sess.opts.static { ret x; }
    let nn2 = default_native_lib_naming(sess, true);
    ret find_library_crate_aux(sess, span, nn2, metas, hash,
                               sess.filesearch);
}

fn default_native_lib_naming(sess: session::session, static: bool) ->
   {prefix: str, suffix: str} {
    if static { ret {prefix: "lib", suffix: ".rlib"}; }
    alt sess.targ_cfg.os {
      session::os_win32 { ret {prefix: "", suffix: ".dll"}; }
      session::os_macos { ret {prefix: "lib", suffix: ".dylib"}; }
      session::os_linux { ret {prefix: "lib", suffix: ".so"}; }
      session::os_freebsd { ret {prefix: "lib", suffix: ".so"}; }
    }
}

fn find_library_crate_aux(sess: session::session,
                          span: span,
                          nn: {prefix: str, suffix: str},
                          metas: [@ast::meta_item],
                          hash: str,
                          filesearch: filesearch::filesearch) ->
   option<{ident: str, data: @[u8]}> {
    let crate_name = crate_name_from_metas(metas);
    let prefix: str = nn.prefix + crate_name + "-";
    let suffix: str = nn.suffix;

    let mut matches = [];
    filesearch::search(filesearch, { |path|
        #debug("inspecting file %s", path);
        let f: str = path::basename(path);
        if !(str::starts_with(f, prefix) && str::ends_with(f, suffix)) {
            #debug("skipping %s, doesn't look like %s*%s", path, prefix,
                   suffix);
            option::none::<()>
        } else {
            #debug("%s is a candidate", path);
            alt get_metadata_section(sess, path) {
              option::some(cvec) {
                if !crate_matches(cvec, metas, hash) {
                    #debug("skipping %s, metadata doesn't match", path);
                    option::none::<()>
                } else {
                    #debug("found %s with matching metadata", path);
                    matches += [{ident: path, data: cvec}];
                    option::none::<()>
                }
              }
              _ {
                #debug("could not load metadata for %s", path);
                option::none::<()>
              }
            }
        }
    });

    if matches.is_empty() {
        none
    } else if matches.len() == 1u {
        some(matches[0])
    } else {
        sess.span_err(
            span, #fmt("multiple matching crates for `%s`", crate_name));
        sess.note("candidates:");
        for matches.each {|match|
            sess.note(#fmt("path: %s", match.ident));
            let attrs = decoder::get_crate_attributes(match.data);
            note_linkage_attrs(sess, attrs);
        }
        sess.abort_if_errors();
        none
    }
}

fn crate_name_from_metas(metas: [@ast::meta_item]) -> str {
    let name_items = attr::find_meta_items_by_name(metas, "name");
    alt vec::last_opt(name_items) {
      some(i) {
        alt attr::get_meta_item_value_str(i) {
          some(n) { n }
          // FIXME: Probably want a warning here since the user
          // is using the wrong type of meta item (#2406)
          _ { fail }
        }
      }
      none { fail "expected to find the crate name" }
    }
}

fn note_linkage_attrs(sess: session::session, attrs: [ast::attribute]) {
    for attr::find_linkage_attrs(attrs).each {|attr|
        sess.note(#fmt("meta: %s", pprust::attr_to_str(attr)));
    }
}

fn crate_matches(crate_data: @[u8], metas: [@ast::meta_item], hash: str) ->
    bool {
    let attrs = decoder::get_crate_attributes(crate_data);
    let linkage_metas = attr::find_linkage_metas(attrs);
    if hash.is_not_empty() {
        let chash = decoder::get_crate_hash(crate_data);
        if chash != hash { ret false; }
    }
    metadata_matches(linkage_metas, metas)
}

fn metadata_matches(extern_metas: [@ast::meta_item],
                    local_metas: [@ast::meta_item]) -> bool {

    #debug("matching %u metadata requirements against %u items",
           vec::len(local_metas), vec::len(extern_metas));

    #debug("crate metadata:");
    for extern_metas.each {|have|
        #debug("  %s", pprust::meta_item_to_str(*have));
    }

    for local_metas.each {|needed|
        #debug("looking for %s", pprust::meta_item_to_str(*needed));
        if !attr::contains(extern_metas, needed) {
            #debug("missing %s", pprust::meta_item_to_str(*needed));
            ret false;
        }
    }
    ret true;
}

fn get_metadata_section(sess: session::session,
                        filename: str) -> option<@[u8]> unsafe {
    let mb = str::as_c_str(filename, {|buf|
        llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(buf)
                                   });
    if mb as int == 0 { ret option::none::<@[u8]>; }
    let of = alt mk_object_file(mb) {
        option::some(of) { of }
        _ { ret option::none::<@[u8]>; }
    };
    let si = mk_section_iter(of.llof);
    while llvm::LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) == False {
        let name_buf = llvm::LLVMGetSectionName(si.llsi);
        let name = unsafe { str::unsafe::from_c_str(name_buf) };
        if str::eq(name, sess.targ_cfg.target_strs.meta_sect_name) {
            let cbuf = llvm::LLVMGetSectionContents(si.llsi);
            let csz = llvm::LLVMGetSectionSize(si.llsi) as uint;
            unsafe {
                let cvbuf: *u8 = unsafe::reinterpret_cast(cbuf);
                ret some(@vec::unsafe::from_buf(cvbuf, csz));
            }
        }
        llvm::LLVMMoveToNextSection(si.llsi);
    }
    ret option::none::<@[u8]>;
}

// A diagnostic function for dumping crate metadata to an output stream
fn list_file_metadata(sess: session::session, path: str, out: io::writer) {
    alt get_metadata_section(sess, path) {
      option::some(bytes) { decoder::list_crate_metadata(bytes, out); }
      option::none {
        out.write_str("could not find metadata in " + path + ".\n");
      }
    }
}
