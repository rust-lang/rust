// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Finds crate binaries and loads their metadata


use lib::llvm::{False, llvm, mk_object_file, mk_section_iter};
use metadata::decoder;
use metadata::encoder;
use metadata::filesearch::FileSearch;
use metadata::filesearch;
use syntax::codemap::span;
use syntax::diagnostic::span_handler;
use syntax::parse::token::ident_interner;
use syntax::print::pprust;
use syntax::{ast, attr};
use syntax::attr::AttrMetaMethods;

use std::c_str::ToCStr;
use std::cast;
use std::io;
use std::num;
use std::option;
use std::os::consts::{macos, freebsd, linux, android, win32};
use std::ptr;
use std::str;
use std::vec;
use extra::flate;

pub enum os {
    os_macos,
    os_win32,
    os_linux,
    os_android,
    os_freebsd
}

pub struct Context {
    diag: @mut span_handler,
    filesearch: @FileSearch,
    span: span,
    ident: @str,
    metas: ~[@ast::MetaItem],
    hash: @str,
    os: os,
    is_static: bool,
    intr: @ident_interner
}

pub fn load_library_crate(cx: &Context) -> (~str, @~[u8]) {
    match find_library_crate(cx) {
      Some(t) => t,
      None => {
        cx.diag.span_fatal(cx.span,
                           fmt!("can't find crate for `%s`",
                                cx.ident));
      }
    }
}

fn find_library_crate(cx: &Context) -> Option<(~str, @~[u8])> {
    attr::require_unique_names(cx.diag, cx.metas);
    find_library_crate_aux(cx, libname(cx), cx.filesearch)
}

fn libname(cx: &Context) -> (~str, ~str) {
    if cx.is_static { return (~"lib", ~".rlib"); }
    let (dll_prefix, dll_suffix) = match cx.os {
        os_win32 => (win32::DLL_PREFIX, win32::DLL_SUFFIX),
        os_macos => (macos::DLL_PREFIX, macos::DLL_SUFFIX),
        os_linux => (linux::DLL_PREFIX, linux::DLL_SUFFIX),
        os_android => (android::DLL_PREFIX, android::DLL_SUFFIX),
        os_freebsd => (freebsd::DLL_PREFIX, freebsd::DLL_SUFFIX),
    };

    (dll_prefix.to_owned(), dll_suffix.to_owned())
}

fn find_library_crate_aux(
    cx: &Context,
    (prefix, suffix): (~str, ~str),
    filesearch: @filesearch::FileSearch
) -> Option<(~str, @~[u8])> {
    let crate_name = crate_name_from_metas(cx.metas);
    // want: crate_name.dir_part() + prefix + crate_name.file_part + "-"
    let prefix = fmt!("%s%s-", prefix, crate_name);
    let mut matches = ~[];
    filesearch::search(filesearch, |path| -> Option<()> {
      let path_str = path.filename();
      match path_str {
          None => None,
          Some(path_str) =>
              if path_str.starts_with(prefix) && path_str.ends_with(suffix) {
                  debug!("%s is a candidate", path.to_str());
                  match get_metadata_section(cx.os, path) {
                      Some(cvec) =>
                          if !crate_matches(cvec, cx.metas, cx.hash) {
                              debug!("skipping %s, metadata doesn't match",
                                  path.to_str());
                              None
                          } else {
                              debug!("found %s with matching metadata", path.to_str());
                              matches.push((path.to_str(), cvec));
                              None
                          },
                      _ => {
                          debug!("could not load metadata for %s", path.to_str());
                          None
                      }
                  }
               }
               else {
                   None
               }
      }
    });

    match matches.len() {
        0 => None,
        1 => Some(matches[0]),
        _ => {
            cx.diag.span_err(
                    cx.span, fmt!("multiple matching crates for `%s`", crate_name));
                cx.diag.handler().note("candidates:");
                for pair in matches.iter() {
                    let ident = pair.first();
                    let data = pair.second();
                    cx.diag.handler().note(fmt!("path: %s", ident));
                    let attrs = decoder::get_crate_attributes(data);
                    note_linkage_attrs(cx.intr, cx.diag, attrs);
                }
                cx.diag.handler().abort_if_errors();
                None
        }
    }
}

pub fn crate_name_from_metas(metas: &[@ast::MetaItem]) -> @str {
    for m in metas.iter() {
        match m.name_str_pair() {
            Some((name, s)) if "name" == name => { return s; }
            _ => {}
        }
    }
    fail!("expected to find the crate name")
}

pub fn package_id_from_metas(metas: &[@ast::MetaItem]) -> Option<@str> {
    for m in metas.iter() {
        match m.name_str_pair() {
            Some((name, s)) if "package_id" == name => { return Some(s); }
            _ => {}
        }
    }
    None
}

pub fn note_linkage_attrs(intr: @ident_interner,
                          diag: @mut span_handler,
                          attrs: ~[ast::Attribute]) {
    let r = attr::find_linkage_metas(attrs);
    for mi in r.iter() {
        diag.handler().note(fmt!("meta: %s", pprust::meta_item_to_str(*mi,intr)));
    }
}

fn crate_matches(crate_data: @~[u8],
                 metas: &[@ast::MetaItem],
                 hash: @str) -> bool {
    let attrs = decoder::get_crate_attributes(crate_data);
    let linkage_metas = attr::find_linkage_metas(attrs);
    if !hash.is_empty() {
        let chash = decoder::get_crate_hash(crate_data);
        if chash != hash { return false; }
    }
    metadata_matches(linkage_metas, metas)
}

pub fn metadata_matches(extern_metas: &[@ast::MetaItem],
                        local_metas: &[@ast::MetaItem]) -> bool {

// extern_metas: metas we read from the crate
// local_metas: metas we're looking for
    debug!("matching %u metadata requirements against %u items",
           local_metas.len(), extern_metas.len());

    do local_metas.iter().all |needed| {
        attr::contains(extern_metas, *needed)
    }
}

fn get_metadata_section(os: os,
                        filename: &Path) -> Option<@~[u8]> {
    unsafe {
        let mb = do filename.to_c_str().with_ref |buf| {
            llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(buf)
        };
        if mb as int == 0 { return option::None::<@~[u8]>; }
        let of = match mk_object_file(mb) {
            option::Some(of) => of,
            _ => return option::None::<@~[u8]>
        };
        let si = mk_section_iter(of.llof);
        while llvm::LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) == False {
            let name_buf = llvm::LLVMGetSectionName(si.llsi);
            let name = str::raw::from_c_str(name_buf);
            debug!("get_metadata_section: name %s", name);
            if name == read_meta_section_name(os) {
                let cbuf = llvm::LLVMGetSectionContents(si.llsi);
                let csz = llvm::LLVMGetSectionSize(si.llsi) as uint;
                let mut found = None;
                let cvbuf: *u8 = cast::transmute(cbuf);
                let vlen = encoder::metadata_encoding_version.len();
                debug!("checking %u bytes of metadata-version stamp",
                       vlen);
                let minsz = num::min(vlen, csz);
                let mut version_ok = false;
                do vec::raw::buf_as_slice(cvbuf, minsz) |buf0| {
                    version_ok = (buf0 ==
                                  encoder::metadata_encoding_version);
                }
                if !version_ok { return None; }

                let cvbuf1 = ptr::offset(cvbuf, vlen as int);
                debug!("inflating %u bytes of compressed metadata",
                       csz - vlen);
                do vec::raw::buf_as_slice(cvbuf1, csz-vlen) |bytes| {
                    let inflated = flate::inflate_bytes(bytes);
                    found = Some(@(inflated));
                }
                if found != None {
                    return found;
                }
            }
            llvm::LLVMMoveToNextSection(si.llsi);
        }
        return option::None::<@~[u8]>;
    }
}

pub fn meta_section_name(os: os) -> ~str {
    match os {
      os_macos => ~"__DATA,__note.rustc",
      os_win32 => ~".note.rustc",
      os_linux => ~".note.rustc",
      os_android => ~".note.rustc",
      os_freebsd => ~".note.rustc"
    }
}

pub fn read_meta_section_name(os: os) -> ~str {
    match os {
      os_macos => ~"__note.rustc",
      os_win32 => ~".note.rustc",
      os_linux => ~".note.rustc",
      os_android => ~".note.rustc",
      os_freebsd => ~".note.rustc"
    }
}

// A diagnostic function for dumping crate metadata to an output stream
pub fn list_file_metadata(intr: @ident_interner,
                          os: os,
                          path: &Path,
                          out: @io::Writer) {
    match get_metadata_section(os, path) {
      option::Some(bytes) => decoder::list_crate_metadata(intr, bytes, out),
      option::None => {
        out.write_str(fmt!("could not find metadata in %s.\n", path.to_str()))
      }
    }
}
