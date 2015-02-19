// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::archive::{Archive, ArchiveBuilder, ArchiveConfig, METADATA_FILENAME};
use super::archive;
use super::rpath;
use super::rpath::RPathConfig;
use super::svh::Svh;
use session::config;
use session::config::NoDebugInfo;
use session::config::{OutputFilenames, Input, OutputTypeBitcode, OutputTypeExe, OutputTypeObject};
use session::search_paths::PathKind;
use session::Session;
use metadata::common::LinkMeta;
use metadata::{encoder, cstore, filesearch, csearch, creader};
use metadata::filesearch::FileDoesntMatch;
use trans::{CrateContext, CrateTranslation, gensym_name};
use middle::ty::{self, Ty};
use util::common::time;
use util::ppaux;
use util::sha2::{Digest, Sha256};

use std::old_io::fs::PathExtensions;
use std::old_io::{fs, TempDir, Command};
use std::old_io;
use std::mem;
use std::str;
use std::string::String;
use flate;
use serialize::hex::ToHex;
use syntax::ast;
use syntax::ast_map::{PathElem, PathElems, PathName};
use syntax::attr::AttrMetaMethods;
use syntax::codemap::Span;
use syntax::parse::token;

// RLIB LLVM-BYTECODE OBJECT LAYOUT
// Version 1
// Bytes    Data
// 0..10    "RUST_OBJECT" encoded in ASCII
// 11..14   format version as little-endian u32
// 15..22   size in bytes of deflate compressed LLVM bitcode as
//          little-endian u64
// 23..     compressed LLVM bitcode

// This is the "magic number" expected at the beginning of a LLVM bytecode
// object in an rlib.
pub const RLIB_BYTECODE_OBJECT_MAGIC: &'static [u8] = b"RUST_OBJECT";

// The version number this compiler will write to bytecode objects in rlibs
pub const RLIB_BYTECODE_OBJECT_VERSION: u32 = 1;

// The offset in bytes the bytecode object format version number can be found at
pub const RLIB_BYTECODE_OBJECT_VERSION_OFFSET: uint = 11;

// The offset in bytes the size of the compressed bytecode can be found at in
// format version 1
pub const RLIB_BYTECODE_OBJECT_V1_DATASIZE_OFFSET: uint =
    RLIB_BYTECODE_OBJECT_VERSION_OFFSET + 4;

// The offset in bytes the compressed LLVM bytecode can be found at in format
// version 1
pub const RLIB_BYTECODE_OBJECT_V1_DATA_OFFSET: uint =
    RLIB_BYTECODE_OBJECT_V1_DATASIZE_OFFSET + 8;


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

pub fn find_crate_name(sess: Option<&Session>,
                       attrs: &[ast::Attribute],
                       input: &Input) -> String {
    let validate = |s: String, span: Option<Span>| {
        creader::validate_crate_name(sess, &s[..], span);
        s
    };

    // Look in attributes 100% of the time to make sure the attribute is marked
    // as used. After doing this, however, we still prioritize a crate name from
    // the command line over one found in the #[crate_name] attribute. If we
    // find both we ensure that they're the same later on as well.
    let attr_crate_name = attrs.iter().find(|at| at.check_name("crate_name"))
                               .and_then(|at| at.value_str().map(|s| (at, s)));

    if let Some(sess) = sess {
        if let Some(ref s) = sess.opts.crate_name {
            if let Some((attr, ref name)) = attr_crate_name {
                if *s != &name[..] {
                    let msg = format!("--crate-name and #[crate_name] are \
                                       required to match, but `{}` != `{}`",
                                      s, name);
                    sess.span_err(attr.span, &msg[..]);
                }
            }
            return validate(s.clone(), None);
        }
    }

    if let Some((attr, s)) = attr_crate_name {
        return validate(s.to_string(), Some(attr.span));
    }
    if let Input::File(ref path) = *input {
        if let Some(s) = path.filestem_str() {
            return validate(s.to_string(), None);
        }
    }

    "rust-out".to_string()
}

pub fn build_link_meta(sess: &Session, krate: &ast::Crate,
                       name: String) -> LinkMeta {
    let r = LinkMeta {
        crate_name: name,
        crate_hash: Svh::calculate(&sess.opts.cg.metadata, krate),
    };
    info!("{:?}", r);
    return r;
}

fn truncated_hash_result(symbol_hasher: &mut Sha256) -> String {
    let output = symbol_hasher.result_bytes();
    // 64 bits should be enough to avoid collisions.
    output[.. 8].to_hex().to_string()
}


// This calculates STH for a symbol, as defined above
fn symbol_hash<'tcx>(tcx: &ty::ctxt<'tcx>,
                     symbol_hasher: &mut Sha256,
                     t: Ty<'tcx>,
                     link_meta: &LinkMeta)
                     -> String {
    // NB: do *not* use abbrevs here as we want the symbol names
    // to be independent of one another in the crate.

    symbol_hasher.reset();
    symbol_hasher.input_str(&link_meta.crate_name[]);
    symbol_hasher.input_str("-");
    symbol_hasher.input_str(link_meta.crate_hash.as_str());
    for meta in &*tcx.sess.crate_metadata.borrow() {
        symbol_hasher.input_str(&meta[..]);
    }
    symbol_hasher.input_str("-");
    symbol_hasher.input_str(&encoder::encoded_ty(tcx, t)[]);
    // Prefix with 'h' so that it never blends into adjacent digits
    let mut hash = String::from_str("h");
    hash.push_str(&truncated_hash_result(symbol_hasher)[]);
    hash
}

fn get_symbol_hash<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>) -> String {
    match ccx.type_hashcodes().borrow().get(&t) {
        Some(h) => return h.to_string(),
        None => {}
    }

    let mut symbol_hasher = ccx.symbol_hasher().borrow_mut();
    let hash = symbol_hash(ccx.tcx(), &mut *symbol_hasher, t, ccx.link_meta());
    ccx.type_hashcodes().borrow_mut().insert(t, hash.clone());
    hash
}


// Name sanitation. LLVM will happily accept identifiers with weird names, but
// gas doesn't!
// gas accepts the following characters in symbols: a-z, A-Z, 0-9, ., _, $
pub fn sanitize(s: &str) -> String {
    let mut result = String::new();
    for c in s.chars() {
        match c {
            // Escape these with $ sequences
            '@' => result.push_str("$SP$"),
            '*' => result.push_str("$BP$"),
            '&' => result.push_str("$RF$"),
            '<' => result.push_str("$LT$"),
            '>' => result.push_str("$GT$"),
            '(' => result.push_str("$LP$"),
            ')' => result.push_str("$RP$"),
            ',' => result.push_str("$C$"),

            // '.' doesn't occur in types and functions, so reuse it
            // for ':' and '-'
            '-' | ':' => result.push('.'),

            // These are legal symbols
            'a' ... 'z'
            | 'A' ... 'Z'
            | '0' ... '9'
            | '_' | '.' | '$' => result.push(c),

            _ => {
                result.push('$');
                for c in c.escape_unicode().skip(1) {
                    match c {
                        '{' => {},
                        '}' => result.push('$'),
                        c => result.push(c),
                    }
                }
            }
        }
    }

    // Underscore-qualify anything that didn't start as an ident.
    if result.len() > 0 &&
        result.as_bytes()[0] != '_' as u8 &&
        ! (result.as_bytes()[0] as char).is_xid_start() {
        return format!("_{}", &result[..]);
    }

    return result;
}

pub fn mangle<PI: Iterator<Item=PathElem>>(path: PI,
                                      hash: Option<&str>) -> String {
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

    let mut n = String::from_str("_ZN"); // _Z == Begin name-sequence, N == nested

    fn push(n: &mut String, s: &str) {
        let sani = sanitize(s);
        n.push_str(&format!("{}{}", sani.len(), sani)[]);
    }

    // First, connect each component with <len, name> pairs.
    for e in path {
        push(&mut n, &token::get_name(e.name()))
    }

    match hash {
        Some(s) => push(&mut n, s),
        None => {}
    }

    n.push('E'); // End name-sequence.
    n
}

pub fn exported_name(path: PathElems, hash: &str) -> String {
    mangle(path, Some(hash))
}

pub fn mangle_exported_name<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, path: PathElems,
                                      t: Ty<'tcx>, id: ast::NodeId) -> String {
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
    hash.push(EXTRA_CHARS.as_bytes()[extra1] as char);
    hash.push(EXTRA_CHARS.as_bytes()[extra2] as char);
    hash.push(EXTRA_CHARS.as_bytes()[extra3] as char);

    exported_name(path, &hash[..])
}

pub fn mangle_internal_name_by_type_and_seq<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                      t: Ty<'tcx>,
                                                      name: &str) -> String {
    let s = ppaux::ty_to_string(ccx.tcx(), t);
    let path = [PathName(token::intern(&s[..])),
                gensym_name(name)];
    let hash = get_symbol_hash(ccx, t);
    mangle(path.iter().cloned(), Some(&hash[..]))
}

pub fn mangle_internal_name_by_path_and_seq(path: PathElems, flav: &str) -> String {
    mangle(path.chain(Some(gensym_name(flav)).into_iter()), None)
}

pub fn get_cc_prog(sess: &Session) -> String {
    match sess.opts.cg.linker {
        Some(ref linker) => return linker.to_string(),
        None => sess.target.target.options.linker.clone(),
    }
}

pub fn remove(sess: &Session, path: &Path) {
    match fs::unlink(path) {
        Ok(..) => {}
        Err(e) => {
            sess.err(&format!("failed to remove {}: {}",
                             path.display(),
                             e)[]);
        }
    }
}

/// Perform the linkage portion of the compilation phase. This will generate all
/// of the requested outputs for this compilation session.
pub fn link_binary(sess: &Session,
                   trans: &CrateTranslation,
                   outputs: &OutputFilenames,
                   crate_name: &str) -> Vec<Path> {
    let mut out_filenames = Vec::new();
    for &crate_type in &*sess.crate_types.borrow() {
        if invalid_output_for_target(sess, crate_type) {
            sess.bug(&format!("invalid output type `{:?}` for target os `{}`",
                             crate_type, sess.opts.target_triple)[]);
        }
        let out_file = link_binary_output(sess, trans, crate_type, outputs,
                                          crate_name);
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


/// Returns default crate type for target
///
/// Default crate type is used when crate type isn't provided neither
/// through cmd line arguments nor through crate attributes
///
/// It is CrateTypeExecutable for all platforms but iOS as there is no
/// way to run iOS binaries anyway without jailbreaking and
/// interaction with Rust code through static library is the only
/// option for now
pub fn default_output_for_target(sess: &Session) -> config::CrateType {
    if !sess.target.target.options.executables {
        config::CrateTypeStaticlib
    } else {
        config::CrateTypeExecutable
    }
}

/// Checks if target supports crate_type as output
pub fn invalid_output_for_target(sess: &Session,
                                 crate_type: config::CrateType) -> bool {
    match (sess.target.target.options.dynamic_linking,
           sess.target.target.options.executables, crate_type) {
        (false, _, config::CrateTypeDylib) => true,
        (_, false, config::CrateTypeExecutable) => true,
        _ => false
    }
}

fn is_writeable(p: &Path) -> bool {
    match p.stat() {
        Err(..) => true,
        Ok(m) => m.perm & old_io::USER_WRITE == old_io::USER_WRITE
    }
}

pub fn filename_for_input(sess: &Session,
                          crate_type: config::CrateType,
                          name: &str,
                          out_filename: &Path) -> Path {
    let libname = format!("{}{}", name, sess.opts.cg.extra_filename);
    match crate_type {
        config::CrateTypeRlib => {
            out_filename.with_filename(format!("lib{}.rlib", libname))
        }
        config::CrateTypeDylib => {
            let (prefix, suffix) = (&sess.target.target.options.dll_prefix[],
                                    &sess.target.target.options.dll_suffix[]);
            out_filename.with_filename(format!("{}{}{}",
                                               prefix,
                                               libname,
                                               suffix))
        }
        config::CrateTypeStaticlib => {
            out_filename.with_filename(format!("lib{}.a", libname))
        }
        config::CrateTypeExecutable => {
            let suffix = &sess.target.target.options.exe_suffix[];
            out_filename.with_filename(format!("{}{}", libname, suffix))
        }
    }
}

fn link_binary_output(sess: &Session,
                      trans: &CrateTranslation,
                      crate_type: config::CrateType,
                      outputs: &OutputFilenames,
                      crate_name: &str) -> Path {
    let obj_filename = outputs.temp_path(OutputTypeObject);
    let out_filename = match outputs.single_output_file {
        Some(ref file) => file.clone(),
        None => {
            let out_filename = outputs.path(OutputTypeExe);
            filename_for_input(sess, crate_type, crate_name, &out_filename)
        }
    };

    // Make sure the output and obj_filename are both writeable.
    // Mac, FreeBSD, and Windows system linkers check this already --
    // however, the Linux linker will happily overwrite a read-only file.
    // We should be consistent.
    let obj_is_writeable = is_writeable(&obj_filename);
    let out_is_writeable = is_writeable(&out_filename);
    if !out_is_writeable {
        sess.fatal(&format!("output file {} is not writeable -- check its \
                            permissions.",
                           out_filename.display())[]);
    }
    else if !obj_is_writeable {
        sess.fatal(&format!("object file {} is not writeable -- check its \
                            permissions.",
                           obj_filename.display())[]);
    }

    match crate_type {
        config::CrateTypeRlib => {
            link_rlib(sess, Some(trans), &obj_filename, &out_filename).build();
        }
        config::CrateTypeStaticlib => {
            link_staticlib(sess, &obj_filename, &out_filename);
        }
        config::CrateTypeExecutable => {
            link_natively(sess, trans, false, &obj_filename, &out_filename);
        }
        config::CrateTypeDylib => {
            link_natively(sess, trans, true, &obj_filename, &out_filename);
        }
    }

    out_filename
}

fn archive_search_paths(sess: &Session) -> Vec<Path> {
    let mut search = Vec::new();
    sess.target_filesearch(PathKind::Native).for_each_lib_search_path(|path, _| {
        search.push(path.clone());
        FileDoesntMatch
    });
    return search;
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
                 out_filename: &Path) -> ArchiveBuilder<'a> {
    let handler = &sess.diagnostic().handler;
    let config = ArchiveConfig {
        handler: handler,
        dst: out_filename.clone(),
        lib_search_paths: archive_search_paths(sess),
        slib_prefix: sess.target.target.options.staticlib_prefix.clone(),
        slib_suffix: sess.target.target.options.staticlib_suffix.clone(),
        maybe_ar_prog: sess.opts.cg.ar.clone()
    };
    let mut ab = ArchiveBuilder::create(config);
    ab.add_file(obj_filename).unwrap();

    for &(ref l, kind) in &*sess.cstore.get_used_libraries().borrow() {
        match kind {
            cstore::NativeStatic => {
                ab.add_native_library(&l[..]).unwrap();
            }
            cstore::NativeFramework | cstore::NativeUnknown => {}
        }
    }

    // After adding all files to the archive, we need to update the
    // symbol table of the archive.
    ab.update_symbols();

    let mut ab = match sess.target.target.options.is_like_osx {
        // For OSX/iOS, we must be careful to update symbols only when adding
        // object files.  We're about to start adding non-object files, so run
        // `ar` now to process the object files.
        true => ab.build().extend(),
        false => ab,
    };

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
            let tmpdir = TempDir::new("rustc").ok().expect("needs a temp dir");
            let metadata = tmpdir.path().join(METADATA_FILENAME);
            match fs::File::create(&metadata).write_all(&trans.metadata[]) {
                Ok(..) => {}
                Err(e) => {
                    sess.err(&format!("failed to write {}: {}",
                                     metadata.display(),
                                     e)[]);
                    sess.abort_if_errors();
                }
            }
            ab.add_file(&metadata).unwrap();
            remove(sess, &metadata);

            // For LTO purposes, the bytecode of this library is also inserted
            // into the archive.  If codegen_units > 1, we insert each of the
            // bitcode files.
            for i in 0..sess.opts.cg.codegen_units {
                // Note that we make sure that the bytecode filename in the
                // archive is never exactly 16 bytes long by adding a 16 byte
                // extension to it. This is to work around a bug in LLDB that
                // would cause it to crash if the name of a file in an archive
                // was exactly 16 bytes.
                let bc_filename = obj_filename.with_extension(&format!("{}.bc", i));
                let bc_deflated_filename = obj_filename.with_extension(
                    &format!("{}.bytecode.deflate", i)[]);

                let bc_data = match fs::File::open(&bc_filename).read_to_end() {
                    Ok(buffer) => buffer,
                    Err(e) => sess.fatal(&format!("failed to read bytecode: {}",
                                                 e)[])
                };

                let bc_data_deflated = match flate::deflate_bytes(&bc_data[..]) {
                    Some(compressed) => compressed,
                    None => sess.fatal(&format!("failed to compress bytecode from {}",
                                               bc_filename.display())[])
                };

                let mut bc_file_deflated = match fs::File::create(&bc_deflated_filename) {
                    Ok(file) => file,
                    Err(e) => {
                        sess.fatal(&format!("failed to create compressed bytecode \
                                            file: {}", e)[])
                    }
                };

                match write_rlib_bytecode_object_v1(&mut bc_file_deflated,
                                                    bc_data_deflated.as_slice()) {
                    Ok(()) => {}
                    Err(e) => {
                        sess.err(&format!("failed to write compressed bytecode: \
                                          {}", e)[]);
                        sess.abort_if_errors()
                    }
                };

                ab.add_file(&bc_deflated_filename).unwrap();
                remove(sess, &bc_deflated_filename);

                // See the bottom of back::write::run_passes for an explanation
                // of when we do and don't keep .0.bc files around.
                let user_wants_numbered_bitcode =
                        sess.opts.output_types.contains(&OutputTypeBitcode) &&
                        sess.opts.cg.codegen_units > 1;
                if !sess.opts.cg.save_temps && !user_wants_numbered_bitcode {
                    remove(sess, &bc_filename);
                }
            }

            // After adding all files to the archive, we need to update the
            // symbol table of the archive. This currently dies on OSX (see
            // #11162), and isn't necessary there anyway
            if !sess.target.target.options.is_like_osx {
                ab.update_symbols();
            }
        }

        None => {}
    }

    ab
}

fn write_rlib_bytecode_object_v1<T: Writer>(writer: &mut T,
                                            bc_data_deflated: &[u8])
                                         -> ::std::old_io::IoResult<()> {
    let bc_data_deflated_size: u64 = bc_data_deflated.len() as u64;

    try! { writer.write_all(RLIB_BYTECODE_OBJECT_MAGIC) };
    try! { writer.write_le_u32(1) };
    try! { writer.write_le_u64(bc_data_deflated_size) };
    try! { writer.write_all(&bc_data_deflated[..]) };

    let number_of_bytes_written_so_far =
        RLIB_BYTECODE_OBJECT_MAGIC.len() +                // magic id
        mem::size_of_val(&RLIB_BYTECODE_OBJECT_VERSION) + // version
        mem::size_of_val(&bc_data_deflated_size) +        // data size field
        bc_data_deflated_size as uint;                    // actual data

    // If the number of bytes written to the object so far is odd, add a
    // padding byte to make it even. This works around a crash bug in LLDB
    // (see issue #15950)
    if number_of_bytes_written_so_far % 2 == 1 {
        try! { writer.write_u8(0) };
    }

    return Ok(());
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
    let ab = link_rlib(sess, None, obj_filename, out_filename);
    let mut ab = match sess.target.target.options.is_like_osx {
        true => ab.build().extend(),
        false => ab,
    };
    if sess.target.target.options.morestack {
        ab.add_native_library("morestack").unwrap();
    }
    if !sess.target.target.options.no_compiler_rt {
        ab.add_native_library("compiler-rt").unwrap();
    }

    let crates = sess.cstore.get_used_crates(cstore::RequireStatic);
    let mut all_native_libs = vec![];

    for &(cnum, ref path) in &crates {
        let ref name = sess.cstore.get_crate_data(cnum).name;
        let p = match *path {
            Some(ref p) => p.clone(), None => {
                sess.err(&format!("could not find rlib for: `{}`",
                                 name)[]);
                continue
            }
        };
        ab.add_rlib(&p, &name[..], sess.lto()).unwrap();

        let native_libs = csearch::get_native_libraries(&sess.cstore, cnum);
        all_native_libs.extend(native_libs.into_iter());
    }

    ab.update_symbols();
    let _ = ab.build();

    if !all_native_libs.is_empty() {
        sess.note("link against the following native artifacts when linking against \
                  this static library");
        sess.note("the order and any duplication can be significant on some platforms, \
                  and so may need to be preserved");
    }

    for &(kind, ref lib) in &all_native_libs {
        let name = match kind {
            cstore::NativeStatic => "static library",
            cstore::NativeUnknown => "library",
            cstore::NativeFramework => "framework",
        };
        sess.note(&format!("{}: {}", name, *lib)[]);
    }
}

// Create a dynamic library or executable
//
// This will invoke the system linker/cc to create the resulting file. This
// links to all upstream files as well.
fn link_natively(sess: &Session, trans: &CrateTranslation, dylib: bool,
                 obj_filename: &Path, out_filename: &Path) {
    let tmpdir = TempDir::new("rustc").ok().expect("needs a temp dir");

    // The invocations of cc share some flags across platforms
    let pname = get_cc_prog(sess);
    let mut cmd = Command::new(&pname[..]);

    cmd.args(&sess.target.target.options.pre_link_args[]);
    link_args(&mut cmd, sess, dylib, tmpdir.path(),
              trans, obj_filename, out_filename);
    cmd.args(&sess.target.target.options.post_link_args[]);
    if !sess.target.target.options.no_compiler_rt {
        cmd.arg("-lcompiler-rt");
    }

    if sess.opts.debugging_opts.print_link_args {
        println!("{:?}", &cmd);
    }

    // May have not found libraries in the right formats.
    sess.abort_if_errors();

    // Invoke the system linker
    debug!("{:?}", &cmd);
    let prog = time(sess.time_passes(), "running linker", (), |()| cmd.output());
    match prog {
        Ok(prog) => {
            if !prog.status.success() {
                sess.err(&format!("linking with `{}` failed: {}",
                                 pname,
                                 prog.status)[]);
                sess.note(&format!("{:?}", &cmd)[]);
                let mut output = prog.error.clone();
                output.push_all(&prog.output[]);
                sess.note(str::from_utf8(&output[..]).unwrap());
                sess.abort_if_errors();
            }
            debug!("linker stderr:\n{}", String::from_utf8(prog.error).unwrap());
            debug!("linker stdout:\n{}", String::from_utf8(prog.output).unwrap());
        },
        Err(e) => {
            sess.err(&format!("could not exec the linker `{}`: {}",
                             pname,
                             e)[]);
            sess.abort_if_errors();
        }
    }


    // On OSX, debuggers need this utility to get run to do some munging of
    // the symbols
    if sess.target.target.options.is_like_osx && sess.opts.debuginfo != NoDebugInfo {
        match Command::new("dsymutil").arg(out_filename).output() {
            Ok(..) => {}
            Err(e) => {
                sess.err(&format!("failed to run dsymutil: {}", e)[]);
                sess.abort_if_errors();
            }
        }
    }
}

fn link_args(cmd: &mut Command,
             sess: &Session,
             dylib: bool,
             tmpdir: &Path,
             trans: &CrateTranslation,
             obj_filename: &Path,
             out_filename: &Path) {

    // The default library location, we need this to find the runtime.
    // The location of crates will be determined as needed.
    let lib_path = sess.target_filesearch(PathKind::All).get_lib_path();

    // target descriptor
    let t = &sess.target.target;

    cmd.arg("-L").arg(&lib_path);

    cmd.arg("-o").arg(out_filename).arg(obj_filename);


    // Stack growth requires statically linking a __morestack function. Note
    // that this is listed *before* all other libraries. Due to the usage of the
    // --as-needed flag below, the standard library may only be useful for its
    // rust_stack_exhausted function. In this case, we must ensure that the
    // libmorestack.a file appears *before* the standard library (so we put it
    // at the very front).
    //
    // Most of the time this is sufficient, except for when LLVM gets super
    // clever. If, for example, we have a main function `fn main() {}`, LLVM
    // will optimize out calls to `__morestack` entirely because the function
    // doesn't need any stack at all!
    //
    // To get around this snag, we specially tell the linker to always include
    // all contents of this library. This way we're guaranteed that the linker
    // will include the __morestack symbol 100% of the time, always resolving
    // references to it even if the object above didn't use it.
    if t.options.morestack {
        if t.options.is_like_osx {
            let morestack = lib_path.join("libmorestack.a");

            let mut v = b"-Wl,-force_load,".to_vec();
            v.push_all(morestack.as_vec());
            cmd.arg(&v[..]);
        } else {
            cmd.args(&["-Wl,--whole-archive", "-lmorestack", "-Wl,--no-whole-archive"]);
        }
    }

    // When linking a dynamic library, we put the metadata into a section of the
    // executable. This metadata is in a separate object file from the main
    // object file, so we link that in here.
    if dylib {
        cmd.arg(obj_filename.with_extension("metadata.o"));
    }

    if t.options.is_like_osx {
        // The dead_strip option to the linker specifies that functions and data
        // unreachable by the entry point will be removed. This is quite useful
        // with Rust's compilation model of compiling libraries at a time into
        // one object file. For example, this brings hello world from 1.7MB to
        // 458K.
        //
        // Note that this is done for both executables and dynamic libraries. We
        // won't get much benefit from dylibs because LLVM will have already
        // stripped away as much as it could. This has not been seen to impact
        // link times negatively.
        //
        // -dead_strip can't be part of the pre_link_args because it's also used for partial
        // linking when using multiple codegen units (-r). So we insert it here.
        cmd.arg("-Wl,-dead_strip");
    }

    // If we're building a dylib, we don't use --gc-sections because LLVM has
    // already done the best it can do, and we also don't want to eliminate the
    // metadata. If we're building an executable, however, --gc-sections drops
    // the size of hello world from 1.8MB to 597K, a 67% reduction.
    if !dylib && !t.options.is_like_osx {
        cmd.arg("-Wl,--gc-sections");
    }

    let used_link_args = sess.cstore.get_used_link_args().borrow();

    if t.options.position_independent_executables {
        let empty_vec = Vec::new();
        let empty_str = String::new();
        let args = sess.opts.cg.link_args.as_ref().unwrap_or(&empty_vec);
        let mut args = args.iter().chain(used_link_args.iter());
        if !dylib
            && (t.options.relocation_model == "pic"
                || *sess.opts.cg.relocation_model.as_ref()
                   .unwrap_or(&empty_str) == "pic")
            && !args.any(|x| *x == "-static") {
            cmd.arg("-pie");
        }
    }

    if t.options.linker_is_gnu {
        // GNU-style linkers support optimization with -O. GNU ld doesn't need a
        // numeric argument, but other linkers do.
        if sess.opts.optimize == config::Default ||
           sess.opts.optimize == config::Aggressive {
            cmd.arg("-Wl,-O1");
        }
    }

    // We want to prevent the compiler from accidentally leaking in any system
    // libraries, so we explicitly ask gcc to not link to any libraries by
    // default. Note that this does not happen for windows because windows pulls
    // in some large number of libraries and I couldn't quite figure out which
    // subset we wanted.
    if !t.options.is_like_windows {
        cmd.arg("-nodefaultlibs");
    }

    // Mark all dynamic libraries and executables as compatible with ASLR
    // FIXME #17098: ASLR breaks gdb
    if t.options.is_like_windows && sess.opts.debuginfo == NoDebugInfo {
        // cmd.arg("-Wl,--dynamicbase");
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
    add_upstream_rust_crates(cmd, sess, dylib, tmpdir, trans);
    add_local_native_libraries(cmd, sess);
    add_upstream_native_libraries(cmd, sess);

    // # Telling the linker what we're doing

    if dylib {
        // On mac we need to tell the linker to let this library be rpathed
        if sess.target.target.options.is_like_osx {
            cmd.args(&["-dynamiclib", "-Wl,-dylib"]);

            if sess.opts.cg.rpath {
                let mut v = "-Wl,-install_name,@rpath/".as_bytes().to_vec();
                v.push_all(out_filename.filename().unwrap());
                cmd.arg(&v[..]);
            }
        } else {
            cmd.arg("-shared");
        }
    }

    // FIXME (#2397): At some point we want to rpath our guesses as to
    // where extern libraries might live, based on the
    // addl_lib_search_paths
    if sess.opts.cg.rpath {
        let sysroot = sess.sysroot();
        let target_triple = &sess.opts.target_triple[];
        let get_install_prefix_lib_path = || {
            let install_prefix = option_env!("CFG_PREFIX").expect("CFG_PREFIX");
            let tlib = filesearch::relative_target_lib_path(sysroot, target_triple);
            let mut path = Path::new(install_prefix);
            path.push(&tlib);

            path
        };
        let rpath_config = RPathConfig {
            used_crates: sess.cstore.get_used_crates(cstore::RequireDynamic),
            out_filename: out_filename.clone(),
            has_rpath: sess.target.target.options.has_rpath,
            is_like_osx: sess.target.target.options.is_like_osx,
            get_install_prefix_lib_path: get_install_prefix_lib_path,
            realpath: ::util::fs::realpath
        };
        cmd.args(&rpath::get_rpath_flags(rpath_config)[]);
    }

    // Finally add all the linker arguments provided on the command line along
    // with any #[link_args] attributes found inside the crate
    let empty = Vec::new();
    cmd.args(&sess.opts.cg.link_args.as_ref().unwrap_or(&empty)[]);
    cmd.args(&used_link_args[..]);
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
fn add_local_native_libraries(cmd: &mut Command, sess: &Session) {
    sess.target_filesearch(PathKind::All).for_each_lib_search_path(|path, k| {
        match k {
            PathKind::Framework => { cmd.arg("-F").arg(path); }
            _ => { cmd.arg("-L").arg(path); }
        }
        FileDoesntMatch
    });

    // Some platforms take hints about whether a library is static or dynamic.
    // For those that support this, we ensure we pass the option if the library
    // was flagged "static" (most defaults are dynamic) to ensure that if
    // libfoo.a and libfoo.so both exist that the right one is chosen.
    let takes_hints = !sess.target.target.options.is_like_osx;

    let libs = sess.cstore.get_used_libraries();
    let libs = libs.borrow();

    let staticlibs = libs.iter().filter_map(|&(ref l, kind)| {
        if kind == cstore::NativeStatic {Some(l)} else {None}
    });
    let others = libs.iter().filter(|&&(_, kind)| {
        kind != cstore::NativeStatic
    });

    // Platforms that take hints generally also support the --whole-archive
    // flag. We need to pass this flag when linking static native libraries to
    // ensure the entire library is included.
    //
    // For more details see #15460, but the gist is that the linker will strip
    // away any unused objects in the archive if we don't otherwise explicitly
    // reference them. This can occur for libraries which are just providing
    // bindings, libraries with generic functions, etc.
    if takes_hints {
        cmd.arg("-Wl,--whole-archive").arg("-Wl,-Bstatic");
    }
    let search_path = archive_search_paths(sess);
    for l in staticlibs {
        if takes_hints {
            cmd.arg(format!("-l{}", l));
        } else {
            // -force_load is the OSX equivalent of --whole-archive, but it
            // involves passing the full path to the library to link.
            let lib = archive::find_library(&l[..],
                                            &sess.target.target.options.staticlib_prefix,
                                            &sess.target.target.options.staticlib_suffix,
                                            &search_path[..],
                                            &sess.diagnostic().handler);
            let mut v = b"-Wl,-force_load,".to_vec();
            v.push_all(lib.as_vec());
            cmd.arg(&v[..]);
        }
    }
    if takes_hints {
        cmd.arg("-Wl,--no-whole-archive").arg("-Wl,-Bdynamic");
    }

    for &(ref l, kind) in others {
        match kind {
            cstore::NativeUnknown => {
                cmd.arg(format!("-l{}", l));
            }
            cstore::NativeFramework => {
                cmd.arg("-framework").arg(&l[..]);
            }
            cstore::NativeStatic => unreachable!(),
        }
    }
}

// # Rust Crate linking
//
// Rust crates are not considered at all when creating an rlib output. All
// dependencies will be linked when producing the final output (instead of
// the intermediate rlib version)
fn add_upstream_rust_crates(cmd: &mut Command, sess: &Session,
                            dylib: bool, tmpdir: &Path,
                            trans: &CrateTranslation) {
    // All of the heavy lifting has previously been accomplished by the
    // dependency_format module of the compiler. This is just crawling the
    // output of that module, adding crates as necessary.
    //
    // Linking to a rlib involves just passing it to the linker (the linker
    // will slurp up the object files inside), and linking to a dynamic library
    // involves just passing the right -l flag.

    let data = if dylib {
        &trans.crate_formats[config::CrateTypeDylib]
    } else {
        &trans.crate_formats[config::CrateTypeExecutable]
    };

    // Invoke get_used_crates to ensure that we get a topological sorting of
    // crates.
    let deps = sess.cstore.get_used_crates(cstore::RequireDynamic);

    for &(cnum, _) in &deps {
        // We may not pass all crates through to the linker. Some crates may
        // appear statically in an existing dylib, meaning we'll pick up all the
        // symbols from the dylib.
        let kind = match data[cnum as uint - 1] {
            Some(t) => t,
            None => continue
        };
        let src = sess.cstore.get_used_crate_source(cnum).unwrap();
        match kind {
            cstore::RequireDynamic => {
                add_dynamic_crate(cmd, sess, src.dylib.unwrap().0)
            }
            cstore::RequireStatic => {
                add_static_crate(cmd, sess, tmpdir, src.rlib.unwrap().0)
            }
        }

    }

    // Converts a library file-stem into a cc -l argument
    fn unlib<'a>(config: &config::Config, stem: &'a [u8]) -> &'a [u8] {
        if stem.starts_with("lib".as_bytes()) && !config.target.options.is_like_windows {
            &stem[3..]
        } else {
            stem
        }
    }

    // Adds the static "rlib" versions of all crates to the command line.
    fn add_static_crate(cmd: &mut Command, sess: &Session, tmpdir: &Path,
                        cratepath: Path) {
        // When performing LTO on an executable output, all of the
        // bytecode from the upstream libraries has already been
        // included in our object file output. We need to modify all of
        // the upstream archives to remove their corresponding object
        // file to make sure we don't pull the same code in twice.
        //
        // We must continue to link to the upstream archives to be sure
        // to pull in native static dependencies. As the final caveat,
        // on Linux it is apparently illegal to link to a blank archive,
        // so if an archive no longer has any object files in it after
        // we remove `lib.o`, then don't link against it at all.
        //
        // If we're not doing LTO, then our job is simply to just link
        // against the archive.
        if sess.lto() {
            let name = cratepath.filename_str().unwrap();
            let name = &name[3..name.len() - 5]; // chop off lib/.rlib
            time(sess.time_passes(),
                 &format!("altering {}.rlib", name)[],
                 (), |()| {
                let dst = tmpdir.join(cratepath.filename().unwrap());
                match fs::copy(&cratepath, &dst) {
                    Ok(..) => {}
                    Err(e) => {
                        sess.err(&format!("failed to copy {} to {}: {}",
                                         cratepath.display(),
                                         dst.display(),
                                         e)[]);
                        sess.abort_if_errors();
                    }
                }
                // Fix up permissions of the copy, as fs::copy() preserves
                // permissions, but the original file may have been installed
                // by a package manager and may be read-only.
                match fs::chmod(&dst, old_io::USER_READ | old_io::USER_WRITE) {
                    Ok(..) => {}
                    Err(e) => {
                        sess.err(&format!("failed to chmod {} when preparing \
                                          for LTO: {}", dst.display(),
                                         e)[]);
                        sess.abort_if_errors();
                    }
                }
                let handler = &sess.diagnostic().handler;
                let config = ArchiveConfig {
                    handler: handler,
                    dst: dst.clone(),
                    lib_search_paths: archive_search_paths(sess),
                    slib_prefix: sess.target.target.options.staticlib_prefix.clone(),
                    slib_suffix: sess.target.target.options.staticlib_suffix.clone(),
                    maybe_ar_prog: sess.opts.cg.ar.clone()
                };
                let mut archive = Archive::open(config);
                archive.remove_file(&format!("{}.o", name)[]);
                let files = archive.files();
                if files.iter().any(|s| s[].ends_with(".o")) {
                    cmd.arg(dst);
                }
            });
        } else {
            cmd.arg(cratepath);
        }
    }

    // Same thing as above, but for dynamic crates instead of static crates.
    fn add_dynamic_crate(cmd: &mut Command, sess: &Session, cratepath: Path) {
        // If we're performing LTO, then it should have been previously required
        // that all upstream rust dependencies were available in an rlib format.
        assert!(!sess.lto());

        // Just need to tell the linker about where the library lives and
        // what its name is
        let dir = cratepath.dirname();
        if !dir.is_empty() { cmd.arg("-L").arg(dir); }

        let mut v = "-l".as_bytes().to_vec();
        v.push_all(unlib(&sess.target, cratepath.filestem().unwrap()));
        cmd.arg(&v[..]);
    }
}

// Link in all of our upstream crates' native dependencies. Remember that
// all of these upstream native dependencies are all non-static
// dependencies. We've got two cases then:
//
// 1. The upstream crate is an rlib. In this case we *must* link in the
// native dependency because the rlib is just an archive.
//
// 2. The upstream crate is a dylib. In order to use the dylib, we have to
// have the dependency present on the system somewhere. Thus, we don't
// gain a whole lot from not linking in the dynamic dependency to this
// crate as well.
//
// The use case for this is a little subtle. In theory the native
// dependencies of a crate are purely an implementation detail of the crate
// itself, but the problem arises with generic and inlined functions. If a
// generic function calls a native function, then the generic function must
// be instantiated in the target crate, meaning that the native symbol must
// also be resolved in the target crate.
fn add_upstream_native_libraries(cmd: &mut Command, sess: &Session) {
    // Be sure to use a topological sorting of crates because there may be
    // interdependencies between native libraries. When passing -nodefaultlibs,
    // for example, almost all native libraries depend on libc, so we have to
    // make sure that's all the way at the right (liblibc is near the base of
    // the dependency chain).
    //
    // This passes RequireStatic, but the actual requirement doesn't matter,
    // we're just getting an ordering of crate numbers, we're not worried about
    // the paths.
    let crates = sess.cstore.get_used_crates(cstore::RequireStatic);
    for (cnum, _) in crates {
        let libs = csearch::get_native_libraries(&sess.cstore, cnum);
        for &(kind, ref lib) in &libs {
            match kind {
                cstore::NativeUnknown => {
                    cmd.arg(format!("-l{}", *lib));
                }
                cstore::NativeFramework => {
                    cmd.arg("-framework");
                    cmd.arg(&lib[..]);
                }
                cstore::NativeStatic => {
                    sess.bug("statics shouldn't be propagated");
                }
            }
        }
    }
}
