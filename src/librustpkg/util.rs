// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::*;
use core::cmp::Ord;
use core::hash::Streaming;
use core::rt::io::Writer;
use rustc::driver::{driver, session};
use rustc::metadata::filesearch;
use std::getopts::groups::getopts;
use std::semver;
use std::term;
use syntax::ast_util::*;
use syntax::codemap::{dummy_sp, spanned, dummy_spanned};
use syntax::ext::base::ExtCtxt;
use syntax::{ast, attr, codemap, diagnostic, fold};
use syntax::ast::{meta_name_value, meta_list};
use syntax::attr::{mk_attr};
use rustc::back::link::output_type_exe;
use rustc::driver::session::{lib_crate, bin_crate};

static Commands: &'static [&'static str] =
    &["build", "clean", "do", "info", "install", "prefer", "test", "uninstall",
      "unprefer"];


pub type ExitCode = int; // For now

/// A version is either an exact revision,
/// or a semantic version
pub enum Version {
    ExactRevision(float),
    SemVersion(semver::Version)
}

impl Ord for Version {
    fn lt(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(f1), &ExactRevision(f2)) => f1 < f2,
            (&SemVersion(ref v1), &SemVersion(ref v2)) => v1 < v2,
            _ => false // incomparable, really
        }
    }
    fn le(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(f1), &ExactRevision(f2)) => f1 <= f2,
            (&SemVersion(ref v1), &SemVersion(ref v2)) => v1 <= v2,
            _ => false // incomparable, really
        }
    }
    fn ge(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(f1), &ExactRevision(f2)) => f1 > f2,
            (&SemVersion(ref v1), &SemVersion(ref v2)) => v1 > v2,
            _ => false // incomparable, really
        }
    }
    fn gt(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(f1), &ExactRevision(f2)) => f1 >= f2,
            (&SemVersion(ref v1), &SemVersion(ref v2)) => v1 >= v2,
            _ => false // incomparable, really
        }
    }

}

impl ToStr for Version {
    fn to_str(&self) -> ~str {
        match *self {
            ExactRevision(ref n) => n.to_str(),
            SemVersion(ref v) => v.to_str()
        }
    }
}

#[deriving(Eq)]
pub enum OutputType { Main, Lib, Bench, Test }

/// Placeholder
pub fn default_version() -> Version { ExactRevision(0.1) }

/// Path-fragment identifier of a package such as
/// 'github.com/graydon/test'; path must be a relative
/// path with >=1 component.
pub struct PkgId {
    /// Remote path: for example, github.com/mozilla/quux-whatever
    remote_path: RemotePath,
    /// Local path: for example, /home/quux/github.com/mozilla/quux_whatever
    /// Note that '-' normalizes to '_' when mapping a remote path
    /// onto a local path
    /// Also, this will change when we implement #6407, though we'll still
    /// need to keep track of separate local and remote paths
    local_path: LocalPath,
    /// Short name. This is the local path's filestem, but we store it
    /// redundantly so as to not call get() everywhere (filestem() returns an
    /// option)
    short_name: ~str,
    version: Version
}

pub impl PkgId {
    fn new(s: &str) -> PkgId {
        use conditions::bad_pkg_id::cond;

        let p = Path(s);
        if p.is_absolute {
            return cond.raise((p, ~"absolute pkgid"));
        }
        if p.components.len() < 1 {
            return cond.raise((p, ~"0-length pkgid"));
        }
        let remote_path = RemotePath(p);
        let local_path = normalize(copy remote_path);
        let short_name = (copy local_path).filestem().expect(fmt!("Strange path! %s", s));
        PkgId {
            local_path: local_path,
            remote_path: remote_path,
            short_name: short_name,
            version: default_version()
        }
    }

    fn hash(&self) -> ~str {
        fmt!("%s-%s-%s", self.remote_path.to_str(),
             hash(self.remote_path.to_str() + self.version.to_str()),
             self.version.to_str())
    }

    fn short_name_with_version(&self) -> ~str {
        fmt!("%s-%s", self.short_name, self.version.to_str())
    }
}

impl ToStr for PkgId {
    fn to_str(&self) -> ~str {
        // should probably use the filestem and not the whole path
        fmt!("%s-%s", self.local_path.to_str(), self.version.to_str())
    }
}

pub struct Pkg {
    id: PkgId,
    bins: ~[~str],
    libs: ~[~str],
}

impl ToStr for Pkg {
    fn to_str(&self) -> ~str {
        self.id.to_str()
    }
}

pub fn root() -> Path {
    match filesearch::get_rustpkg_root() {
        result::Ok(path) => path,
        result::Err(err) => fail!(err)
    }
}

pub fn is_cmd(cmd: &str) -> bool {
    Commands.any(|&c| c == cmd)
}

struct ListenerFn {
    cmds: ~[~str],
    span: codemap::span,
    path: ~[ast::ident]
}

struct ReadyCtx {
    sess: session::Session,
    crate: @ast::crate,
    ext_cx: @ExtCtxt,
    path: ~[ast::ident],
    fns: ~[ListenerFn]
}

fn fold_mod(_ctx: @mut ReadyCtx,
            m: &ast::_mod,
            fold: @fold::ast_fold) -> ast::_mod {
    fn strip_main(item: @ast::item) -> @ast::item {
        @ast::item {
            attrs: do item.attrs.filtered |attr| {
                *attr::get_attr_name(attr) != ~"main"
            },
            .. copy *item
        }
    }

    fold::noop_fold_mod(&ast::_mod {
        items: do m.items.map |item| {
            strip_main(*item)
        },
        .. copy *m
    }, fold)
}

fn fold_item(ctx: @mut ReadyCtx,
             item: @ast::item,
             fold: @fold::ast_fold) -> Option<@ast::item> {
    ctx.path.push(item.ident);

    let attrs = attr::find_attrs_by_name(item.attrs, "pkg_do");

    if attrs.len() > 0 {
        let mut cmds = ~[];

        for attrs.each |attr| {
            match attr.node.value.node {
                ast::meta_list(_, ref mis) => {
                    for mis.each |mi| {
                        match mi.node {
                            ast::meta_word(cmd) => cmds.push(copy *cmd),
                            _ => {}
                        };
                    }
                }
                _ => cmds.push(~"build")
            };
        }

        ctx.fns.push(ListenerFn {
            cmds: cmds,
            span: item.span,
            path: /*bad*/copy ctx.path
        });
    }

    let res = fold::noop_fold_item(item, fold);

    ctx.path.pop();

    res
}

/// Generate/filter main function, add the list of commands, etc.
pub fn ready_crate(sess: session::Session,
                   crate: @ast::crate) -> @ast::crate {
    let ctx = @mut ReadyCtx {
        sess: sess,
        crate: crate,
        ext_cx: ExtCtxt::new(sess.parse_sess, copy sess.opts.cfg),
        path: ~[],
        fns: ~[]
    };
    let precursor = @fold::AstFoldFns {
        // fold_crate: fold::wrap(|a, b| fold_crate(ctx, a, b)),
        fold_item: |a, b| fold_item(ctx, a, b),
        fold_mod: |a, b| fold_mod(ctx, a, b),
        .. *fold::default_ast_fold()
    };

    let fold = fold::make_fold(precursor);

    @fold.fold_crate(crate)
}

pub fn parse_vers(vers: ~str) -> result::Result<semver::Version, ~str> {
    match semver::parse(vers) {
        Some(vers) => result::Ok(vers),
        None => result::Err(~"could not parse version: invalid")
    }
}

pub fn need_dir(s: &Path) {
    if !os::path_is_dir(s) && !os::make_dir(s, 493_i32) {
        fail!("can't create dir: %s", s.to_str());
    }
}

pub fn note(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_green);
        out.write_str("note: ");
        term::reset(out);
        out.write_line(msg);
    } else {
        out.write_line(~"note: " + msg);
    }
}

pub fn warn(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_yellow);
        out.write_str("warning: ");
        term::reset(out);
        out.write_line(msg);
    } else {
        out.write_line(~"warning: " + msg);
    }
}

pub fn error(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_red);
        out.write_str("error: ");
        term::reset(out);
        out.write_line(msg);
    } else {
        out.write_line(~"error: " + msg);
    }
}

pub fn hash(data: ~str) -> ~str {
    let mut hasher = hash::default_state();
    let buffer = str::as_bytes_slice(data);
    hasher.write(buffer);
    hasher.result_str()
}

// FIXME (#4432): Use workcache to only compile when needed
pub fn compile_input(sysroot: Option<@Path>,
                     pkg_id: &PkgId,
                     in_file: &Path,
                     out_dir: &Path,
                     flags: &[~str],
                     cfgs: &[~str],
                     opt: bool,
                     what: OutputType) -> bool {

    assert!(in_file.components.len() > 1);
    let input = driver::file_input(copy *in_file);
    debug!("compile_input: %s / %?", in_file.to_str(), what);
    // tjc: by default, use the package ID name as the link name
    // not sure if we should support anything else

    let binary = @(copy os::args()[0]);
    let building_library = what == Lib;

    let out_file = if building_library {
        out_dir.push(os::dll_filename(pkg_id.short_name))
    }
    else {
        out_dir.push(pkg_id.short_name + match what {
            Test => ~"test", Bench => ~"bench", Main | Lib => ~""
        } + os::EXE_SUFFIX)
    };

    debug!("compiling %s into %s",
           in_file.to_str(),
           out_file.to_str());
    debug!("flags: %s", str::connect(flags, " "));
    debug!("cfgs: %s", str::connect(cfgs, " "));
    debug!("compile_input's sysroot = %?", sysroot);

    let crate_type = match what {
        Lib => lib_crate,
        Test | Bench | Main => bin_crate
    };
    let matches = getopts(~[~"-Z", ~"time-passes"]
                          + match what {
                              Lib => ~[~"--lib"],
                              // --test compiles both #[test] and #[bench] fns
                              Test | Bench => ~[~"--test"],
                              Main => ~[]
                          }
                          + flags
                          + cfgs.flat_map(|&c| { ~[~"--cfg", c] }),
                          driver::optgroups()).get();
    let mut options = session::options {
        crate_type: crate_type,
        optimize: if opt { session::Aggressive } else { session::No },
        test: what == Test || what == Bench,
        maybe_sysroot: sysroot,
        addl_lib_search_paths: ~[copy *out_dir],
        // output_type should be conditional
        output_type: output_type_exe, // Use this to get a library? That's weird
        .. copy *driver::build_session_options(binary, &matches, diagnostic::emit)
    };

    for cfgs.each |&cfg| {
        options.cfg.push(attr::mk_word_item(@cfg));
    }

    let sess = driver::build_session(@options, diagnostic::emit);

    debug!("calling compile_crate_from_input, out_dir = %s,
           building_library = %?", out_dir.to_str(), sess.building_library);
    let _ = compile_crate_from_input(&input, pkg_id, Some(copy *out_dir), sess,
                                     None, &out_file, binary,
                                     driver::cu_everything);
    true
}

// Should use workcache to avoid recompiling when not necessary
// Should also rename this to something better
// If crate_opt is present, then finish compilation. If it's None, then
// call compile_upto and return the crate
// also, too many arguments
pub fn compile_crate_from_input(input: &driver::input,
                                pkg_id: &PkgId,
                                build_dir_opt: Option<Path>,
                                sess: session::Session,
                                crate_opt: Option<@ast::crate>,
                                out_file: &Path,
                                binary: @~str,
                                what: driver::compile_upto) -> @ast::crate {
    debug!("Calling build_output_filenames with %? and %s", build_dir_opt, out_file.to_str());
    let outputs = driver::build_output_filenames(input, &build_dir_opt,
                                                 &Some(copy *out_file), sess);
    debug!("Outputs are %? and output type = %?", outputs, sess.opts.output_type);
    let cfg = driver::build_configuration(sess, binary, input);
    match crate_opt {
        Some(c) => {
            debug!("Calling compile_rest, outputs = %?", outputs);
            assert_eq!(what, driver::cu_everything);
            driver::compile_rest(sess, cfg, driver::cu_everything, Some(outputs), Some(c));
            c
        }
        None => {
            debug!("Calling compile_upto, outputs = %?", outputs);
            let (crate, _) = driver::compile_upto(sess, copy cfg, input,
                                                  driver::cu_parse, Some(outputs));
            let mut crate = crate;

            debug!("About to inject link_meta info...");
            // Inject the inferred link_meta info if it's not already there
            // (assumes that name and vers are the only linkage metas)

            debug!("How many attrs? %?", attr::find_linkage_metas(crate.node.attrs).len());

            if attr::find_linkage_metas(crate.node.attrs).is_empty() {
                crate = @codemap::respan(crate.span, ast::crate_ {
                    attrs: ~[mk_attr(@dummy_spanned(
                        meta_list(@~"link",
                                  ~[@dummy_spanned(meta_name_value(@~"name",
                                        mk_string_lit(@(copy pkg_id.short_name)))),
                                    @dummy_spanned(meta_name_value(@~"vers",
                                        mk_string_lit(@(copy pkg_id.version.to_str()))))])))],
                    ..copy crate.node});
            }

            driver::compile_rest(sess, cfg, what, Some(outputs), Some(crate));
            crate
        }
    }
}

#[cfg(windows)]
pub fn exe_suffix() -> ~str { ~".exe" }

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
pub fn exe_suffix() -> ~str { ~"" }

// Called by build_crates
// FIXME (#4432): Use workcache to only compile when needed
pub fn compile_crate(sysroot: Option<@Path>, pkg_id: &PkgId,
                     crate: &Path, dir: &Path,
                     flags: &[~str], cfgs: &[~str], opt: bool,
                     what: OutputType) -> bool {
    debug!("compile_crate: crate=%s, dir=%s", crate.to_str(), dir.to_str());
    debug!("compile_crate: short_name = %s, flags =...", pkg_id.to_str());
    for flags.each |&fl| {
        debug!("+++ %s", fl);
    }
    compile_input(sysroot, pkg_id, crate, dir, flags, cfgs, opt, what)
}

// normalize should be the only way to construct a LocalPath
// (though this isn't enforced)
/// Replace all occurrences of '-' in the stem part of path with '_'
/// This is because we treat rust-foo-bar-quux and rust_foo_bar_quux
/// as the same name
pub fn normalize(p_: RemotePath) -> LocalPath {
    let RemotePath(p) = p_;
    match p.filestem() {
        None => LocalPath(p),
        Some(st) => {
            let replaced = str::replace(st, "-", "_");
            if replaced != st {
                LocalPath(p.with_filestem(replaced))
            }
            else {
                LocalPath(p)
            }
        }
    }
}

#[cfg(windows)]
pub fn link_exe(_src: &Path, _dest: &Path) -> bool {
    /* FIXME (#1768): Investigate how to do this on win32
       Node wraps symlinks by having a .bat,
       but that won't work with minGW. */

    false
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
pub fn link_exe(src: &Path, dest: &Path) -> bool {
    unsafe {
        do str::as_c_str(src.to_str()) |src_buf| {
            do str::as_c_str(dest.to_str()) |dest_buf| {
                libc::link(src_buf, dest_buf) == 0 as libc::c_int &&
                    libc::chmod(dest_buf, 755) == 0 as libc::c_int
            }
        }
    }
}

pub fn mk_string_lit(s: @~str) -> ast::lit {
    spanned {
        node: ast::lit_str(s),
        span: dummy_sp()
    }
}

/// Wrappers to prevent local and remote paths from getting confused
pub struct RemotePath (Path);
pub struct LocalPath (Path);

#[cfg(test)]
mod test {
    use super::is_cmd;

    #[test]
    fn test_is_cmd() {
        assert!(is_cmd(~"build"));
        assert!(is_cmd(~"clean"));
        assert!(is_cmd(~"do"));
        assert!(is_cmd(~"info"));
        assert!(is_cmd(~"install"));
        assert!(is_cmd(~"prefer"));
        assert!(is_cmd(~"test"));
        assert!(is_cmd(~"uninstall"));
        assert!(is_cmd(~"unprefer"));
    }

}
