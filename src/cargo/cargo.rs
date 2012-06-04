// cargo.rs - Rust package manager

#[warn(no_non_implicitly_copyable_typarams)];

import syntax::{ast, codemap};
import syntax::parse;
import rustc::metadata::filesearch::{get_cargo_root, get_cargo_root_nearest,
                                     get_cargo_sysroot, libdir};
import syntax::diagnostic;

import result::{ok, err};
import io::writer_util;
import result;
import std::{map, json, tempfile, term, sort, getopts};
import map::hashmap;
import str;
import vec;
import getopts::{optflag, optopt, opt_present};

type package = {
    name: str,
    uuid: str,
    url: str,
    method: str,
    description: str,
    ref: option<str>,
    tags: [str]
};

type source = {
    name: str,
    url: str,
    sig: option<str>,
    key: option<str>,
    keyfp: option<str>,
    mut packages: [package]
};

type cargo = {
    pgp: bool,
    root: str,
    installdir: str,
    bindir: str,
    libdir: str,
    workdir: str,
    sourcedir: str,
    sources: map::hashmap<str, source>,
    opts: options
};

type pkg = {
    name: str,
    vers: str,
    uuid: str,
    desc: option<str>,
    sigs: option<str>,
    crate_type: option<str>
};

type options = {
    test: bool,
    mode: mode,
    free: [str],
    help: bool,
};

enum mode { system_mode, user_mode, local_mode }

fn opts() -> [getopts::opt] {
    [optflag("g"), optflag("G"), optflag("test"),
     optflag("h"), optflag("help")]
}

fn info(msg: str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_green);
        out.write_str("info: ");
        term::reset(out);
        out.write_line(msg);
    } else { out.write_line("info: " + msg); }
}

fn warn(msg: str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_yellow);
        out.write_str("warning: ");
        term::reset(out);
        out.write_line(msg);
    }else { out.write_line("warning: " + msg); }
}

fn error(msg: str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_red);
        out.write_str("error: ");
        term::reset(out);
        out.write_line(msg);
    }
    else { out.write_line("error: " + msg); }
}

fn is_uuid(id: str) -> bool {
    let parts = str::split_str(id, "-");
    if vec::len(parts) == 5u {
        let mut correct = 0u;
        for vec::eachi(parts) { |i, part|

            if !part.all(is_hex_digit) {
                ret false;
            }

            fn is_hex_digit(ch: char) -> bool {
                ('0' <= ch && ch <= '9') ||
                    ('a' <= ch && ch <= 'f') ||
                    ('A' <= ch && ch <= 'F')
            }

            alt i {
                0u {
                    if str::len(part) == 8u {
                        correct += 1u;
                    }
                }
                1u | 2u | 3u {
                    if str::len(part) == 4u {
                        correct += 1u;
                    }
                }
                4u {
                    if str::len(part) == 12u {
                        correct += 1u;
                    }
                }
                _ { }
            }
        }
        if correct >= 5u {
            ret true;
        }
    }
    ret false;
}

#[test]
fn test_is_uuid() {
    assert is_uuid("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaafAF09");
    assert !is_uuid("aaaaaaaa-aaaa-aaaa-aaaaa-aaaaaaaaaaaa");
    assert !is_uuid("");
    assert !is_uuid("aaaaaaaa-aaa -aaaa-aaaa-aaaaaaaaaaaa");
    assert !is_uuid("aaaaaaaa-aaa!-aaaa-aaaa-aaaaaaaaaaaa");
    assert !is_uuid("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa-a");
    assert !is_uuid("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaป");
}

// FIXME: implement URI/URL parsing so we don't have to resort to weak checks

fn is_archive_uri(uri: str) -> bool {
    str::ends_with(uri, ".tar")
    || str::ends_with(uri, ".tar.gz")
    || str::ends_with(uri, ".tar.xz")
    || str::ends_with(uri, ".tar.bz2")
}

fn is_archive_url(url: str) -> bool {
    // FIXME: this requires the protocol bit - if we had proper URI parsing,
    // we wouldn't need it

    alt str::find_str(url, "://") {
        option::some(idx) {
            str::ends_with(url, ".tar")
            || str::ends_with(url, ".tar.gz")
            || str::ends_with(url, ".tar.xz")
            || str::ends_with(url, ".tar.bz2")
        }
        option::none { false }
    }
}

fn is_git_url(url: str) -> bool {
    if str::ends_with(url, "/") { str::ends_with(url, ".git/") }
    else {
        str::starts_with(url, "git://") || str::ends_with(url, ".git")
    }
}

fn load_link(mis: [@ast::meta_item]) -> (option<str>,
                                         option<str>,
                                         option<str>) {
    let mut name = none;
    let mut vers = none;
    let mut uuid = none;
    for mis.each {|a|
        alt a.node {
            ast::meta_name_value(v, {node: ast::lit_str(s), span: _}) {
                alt v {
                    "name" { name = some(s); }
                    "vers" { vers = some(s); }
                    "uuid" { uuid = some(s); }
                    _ { }
                }
            }
            _ { fail "load_link: meta items must be name-values"; }
        }
    }
    (name, vers, uuid)
}

fn load_pkg(filename: str) -> option<pkg> {
    let cm = codemap::new_codemap();
    let handler = diagnostic::mk_handler(none);
    let sess = @{
        cm: cm,
        mut next_id: 1,
        span_diagnostic: diagnostic::mk_span_handler(handler, cm),
        mut chpos: 0u,
        mut byte_pos: 0u
    };
    let c = parse::parse_crate_from_crate_file(filename, [], sess);

    let mut name = none;
    let mut vers = none;
    let mut uuid = none;
    let mut desc = none;
    let mut sigs = none;
    let mut crate_type = none;

    for c.node.attrs.each {|a|
        alt a.node.value.node {
            ast::meta_name_value(v, {node: ast::lit_str(s), span: _}) {
                alt v {
                    "desc" { desc = some(v); }
                    "sigs" { sigs = some(v); }
                    "crate_type" { crate_type = some(v); }
                    _ { }
                }
            }
            ast::meta_list(v, mis) {
                if v == "link" {
                    let (n, v, u) = load_link(mis);
                    name = n;
                    vers = v;
                    uuid = u;
                }
            }
            _ { fail "load_pkg: pkg attributes may not contain meta_words"; }
        }
    }

    alt (name, vers, uuid) {
        (some(name0), some(vers0), some(uuid0)) {
            some({
                name: name0,
                vers: vers0,
                uuid: uuid0,
                desc: desc,
                sigs: sigs,
                crate_type: crate_type})
        }
        _ { ret none; }
    }
}

fn print(s: str) {
    io::stdout().write_line(s);
}

fn rest(s: str, start: uint) -> str {
    if (start >= str::len(s)) {
        ""
    } else {
        str::slice(s, start, str::len(s))
    }
}

fn need_dir(s: str) {
    if os::path_is_dir(s) { ret; }
    if !os::make_dir(s, 493_i32 /* oct: 755 */) {
        fail #fmt["can't make_dir %s", s];
    }
}

fn parse_source(name: str, j: json::json) -> source {
    alt j {
        json::dict(_j) {
            let url = alt _j.find("url") {
                some(json::string(u)) {
                    u
                }
                _ { fail "needed 'url' field in source"; }
            };
            let sig = alt _j.find("sig") {
                some(json::string(u)) {
                    some(u)
                }
                _ { none }
            };
            let key = alt _j.find("key") {
                some(json::string(u)) {
                    some(u)
                }
                _ { none }
            };
            let keyfp = alt _j.find("keyfp") {
                some(json::string(u)) {
                    some(u)
                }
                _ { none }
            };
            ret { name: name, url: url, sig: sig, key: key, keyfp: keyfp,
                  mut packages: [] };
        }
        _ { fail "needed dict value in source"; }
    };
}

fn try_parse_sources(filename: str, sources: map::hashmap<str, source>) {
    if !os::path_exists(filename)  { ret; }
    let c = io::read_whole_file_str(filename);
    alt json::from_str(result::get(c)) {
        ok(json::dict(j)) {
            for j.each { |k, v|
                sources.insert(k, parse_source(k, v));
                #debug("source: %s", k);
            }
        }
        ok(_) { fail "malformed sources.json"; }
        err(e) { fail #fmt("%s:%u:%u: %s", filename, e.line, e.col, e.msg); }
    }
}

fn load_one_source_package(&src: source, p: map::hashmap<str, json::json>) {
    let name = alt p.find("name") {
        some(json::string(_n)) { _n }
        _ {
            warn("malformed source json: " + src.name + " (missing name)");
            ret;
        }
    };

    let uuid = alt p.find("uuid") {
        some(json::string(_n)) { _n }
        _ {
            warn("malformed source json: " + src.name + " (missing uuid)");
            ret;
        }
    };

    let url = alt p.find("url") {
        some(json::string(_n)) { _n }
        _ {
            warn("malformed source json: " + src.name + " (missing url)");
            ret;
        }
    };

    let method = alt p.find("method") {
        some(json::string(_n)) { _n }
        _ {
            warn("malformed source json: " + src.name + " (missing method)");
            ret;
        }
    };

    let ref = alt p.find("ref") {
        some(json::string(_n)) { some(_n) }
        _ { none }
    };

    let mut tags = [];
    alt p.find("tags") {
        some(json::list(js)) {
            for js.each {|j|
                alt j {
                    json::string(_j) { vec::grow(tags, 1u, _j); }
                    _ { }
                }
            }
        }
        _ { }
    }

    let description = alt p.find("description") {
        some(json::string(_n)) { _n }
        _ {
            warn("malformed source json: " + src.name
                 + " (missing description)");
            ret;
        }
    };

    vec::grow(src.packages, 1u, {
        name: name,
        uuid: uuid,
        url: url,
        method: method,
        description: description,
        ref: ref,
        tags: tags
    });
    log(debug, "  loaded package: " + src.name + "/" + name);
}

fn load_source_packages(&c: cargo, &src: source) {
    log(debug, "loading source: " + src.name);
    let dir = path::connect(c.sourcedir, src.name);
    let pkgfile = path::connect(dir, "packages.json");
    if !os::path_exists(pkgfile) { ret; }
    let pkgstr = io::read_whole_file_str(pkgfile);
    alt json::from_str(result::get(pkgstr)) {
        ok(json::list(js)) {
            for js.each {|_j|
                alt _j {
                    json::dict(_p) {
                        load_one_source_package(src, _p);
                    }
                    _ {
                        warn("malformed source json: " + src.name +
                             " (non-dict pkg)");
                    }
                }
            }
        }
        ok(_) {
            warn("malformed source json: " + src.name +
                 "(packages is not a list)");
        }
        err(e) {
            warn(#fmt("%s:%u:%u: %s", src.name, e.line, e.col, e.msg));
        }
    };
}

fn build_cargo_options(argv: [str]) -> options {
    let match = alt getopts::getopts(argv, opts()) {
        result::ok(m) { m }
        result::err(f) {
            fail #fmt["%s", getopts::fail_str(f)];
        }
    };

    let test = opt_present(match, "test");
    let G    = opt_present(match, "G");
    let g    = opt_present(match, "g");
    let help = opt_present(match, "h") || opt_present(match, "help");
    let len  = vec::len(match.free);

    let is_install = len > 1u && match.free[1] == "install";
    let is_uninstall = len > 1u && match.free[1] == "uninstall";

    if G && g { fail "-G and -g both provided"; }

    if !is_install && !is_uninstall && (g || G) {
        fail "-g and -G are only valid for `install` and `uninstall|rm`";
    }

    let mode =
        if (!is_install && !is_uninstall) || g { user_mode }
        else if G { system_mode }
        else { local_mode };

    {test: test, mode: mode, free: match.free, help: help}
}

fn configure(opts: options) -> cargo {
    let home = alt get_cargo_root() {
        ok(_home) { _home }
        err(_err) { result::get(get_cargo_sysroot()) }
    };

    let get_cargo_dir = alt opts.mode {
        system_mode { get_cargo_sysroot }
        user_mode { get_cargo_root }
        local_mode { get_cargo_root_nearest }
    };

    let p = result::get(get_cargo_dir());

    let sources = map::str_hash::<source>();
    try_parse_sources(path::connect(home, "sources.json"), sources);
    try_parse_sources(path::connect(home, "local-sources.json"), sources);

    let mut c = {
        pgp: pgp::supported(),
        root: home,
        installdir: p,
        bindir: path::connect(p, "bin"),
        libdir: path::connect(p, "lib"),
        workdir: path::connect(home, "work"),
        sourcedir: path::connect(home, "sources"),
        sources: sources,
        opts: opts
    };

    need_dir(c.root);
    need_dir(c.sourcedir);
    need_dir(c.workdir);
    need_dir(c.installdir);
    need_dir(c.libdir);
    need_dir(c.bindir);

    for sources.each_key { |k|
        let mut s = sources.get(k);
        load_source_packages(c, s);
        sources.insert(k, s);
    }

    if c.pgp {
        pgp::init(c.root);
    } else {
        warn("command `gpg` was not found");
        warn("you have to install gpg from source " +
             " or package manager to get it to work correctly");
    }

    c
}

fn for_each_package(c: cargo, b: fn(source, package)) {
    for c.sources.each_value {|v|
        // FIXME (#2280): this temporary shouldn't be
        // necessary, but seems to be, for borrowing.
        let pks = copy v.packages;
        for vec::each(pks) {|p|
            b(v, p);
        }
    }
}

// Runs all programs in directory <buildpath>
fn run_programs(buildpath: str) {
    let newv = os::list_dir_path(buildpath);
    for newv.each {|ct|
        run::run_program(ct, []);
    }
}

// Runs rustc in <path + subdir> with the given flags
// and returns <path + subdir>
fn run_in_buildpath(what: str, path: str, subdir: str, cf: str,
                    extra_flags: [str]) -> option<str> {
    let buildpath = path::connect(path, subdir);
    need_dir(buildpath);
    #debug("%s: %s -> %s", what, cf, buildpath);
    let p = run::program_output(rustc_sysroot(),
                                ["--out-dir", buildpath, cf] + extra_flags);
    if p.status != 0 {
        error(#fmt["rustc failed: %d\n%s\n%s", p.status, p.err, p.out]);
        ret none;
    }
    some(buildpath)
}

fn test_one_crate(_c: cargo, path: str, cf: str) {
  let buildpath = alt run_in_buildpath("testing", path, "/test", cf,
                                       [ "--test"]) {
      none { ret; }
      some(bp) { bp }
  };
  run_programs(buildpath);
}

fn install_one_crate(c: cargo, path: str, cf: str) {
    let buildpath = alt run_in_buildpath("installing", path,
                                         "/build", cf, []) {
      none { ret; }
      some(bp) { bp }
    };
    let newv = os::list_dir_path(buildpath);
    let exec_suffix = os::exe_suffix();
    for newv.each {|ct|
        if (exec_suffix != "" && str::ends_with(ct, exec_suffix)) ||
            (exec_suffix == "" && !str::starts_with(path::basename(ct),
                                                    "lib")) {
            #debug("  bin: %s", ct);
            install_to_dir(ct, c.bindir);
            if c.opts.mode == system_mode {
                // TODO: Put this file in PATH / symlink it so it can be
                // used as a generic executable
                // `cargo install -G rustray` and `rustray file.obj`
            }
        } else {
            #debug("  lib: %s", ct);
            install_to_dir(ct, c.libdir);
        }
    }
}


fn rustc_sysroot() -> str {
    alt os::self_exe_path() {
        some(_path) {
            let path = [_path, "..", "bin", "rustc"];
            check vec::is_not_empty(path);
            let rustc = path::normalize(path::connect_many(path));
            #debug("  rustc: %s", rustc);
            rustc
        }
        none { "rustc" }
    }
}

fn install_source(c: cargo, path: str) {
    #debug("source: %s", path);
    os::change_dir(path);

    let mut cratefiles = [];
    for os::walk_dir(".") {|p|
        if str::ends_with(p, ".rc") {
            cratefiles += [p];
        }
    }

    if vec::is_empty(cratefiles) {
        fail "this doesn't look like a rust package (no .rc files)";
    }

    for cratefiles.each {|cf|
        let p = load_pkg(cf);
        alt p {
            none { cont; }
            some(_) {
                if c.opts.test {
                    test_one_crate(c, path, cf);
                }
                install_one_crate(c, path, cf);
            }
        }
    }
}

fn install_git(c: cargo, wd: str, url: str, ref: option<str>) {
    info("installing with git from " + url + "...");
    run::run_program("git", ["clone", url, wd]);
    if option::is_some::<str>(ref) {
        let r = option::get::<str>(ref);
        os::change_dir(wd);
        run::run_program("git", ["checkout", r]);
    }

    install_source(c, wd);
}

fn install_curl(c: cargo, wd: str, url: str) {
    info("installing with curl from " + url + "...");
    let tarpath = path::connect(wd, "pkg.tar");
    let p = run::program_output("curl", ["-f", "-s", "-o",
                                         tarpath, url]);
    if p.status != 0 {
        fail #fmt["fetch of %s failed: %s", url, p.err];
    }
    run::run_program("tar", ["-x", "--strip-components=1",
                             "-C", wd, "-f", tarpath]);
    install_source(c, wd);
}

fn install_file(c: cargo, wd: str, path: str) {
    info("installing with tar from " + path + "...");
    run::run_program("tar", ["-x", "--strip-components=1",
                             "-C", wd, "-f", path]);
    install_source(c, wd);
}

fn install_package(c: cargo, wd: str, pkg: package) {
    alt pkg.method {
        "git" { install_git(c, wd, pkg.url, pkg.ref); }
        "http" | "ftp" | "curl" { install_curl(c, wd, pkg.url); }
        "file" { install_file(c, wd, pkg.url); }
        _ { fail #fmt("don't know how to install with: %s", pkg.method) }
    }
}

fn cargo_suggestion(c: cargo, syncing: bool, fallback: fn())
{
    if c.sources.size() == 0u {
        error("no sources defined - you may wish to run " +
              "`cargo init` then `cargo sync`");
        ret;
    }
    if !syncing {
        let mut npkg = 0u;
        for c.sources.each_value { |v| npkg += vec::len(v.packages) }
        if npkg == 0u {
            error("no packages synced - you may wish to run " +
                  "`cargo sync`");
            ret;
        }
    }
    fallback();
}

fn install_uuid(c: cargo, wd: str, uuid: str) {
    let mut ps = [];
    for_each_package(c, { |s, p|
        if p.uuid == uuid {
            vec::grow(ps, 1u, (s.name, p));
        }
    });
    if vec::len(ps) == 1u {
        let (_, p) = ps[0];
        install_package(c, wd, p);
        ret;
    } else if vec::len(ps) == 0u {
        cargo_suggestion(c, false, { ||
            error("can't find package: " + uuid);
        });
        ret;
    }
    error("found multiple packages:");
    for ps.each {|elt|
        let (sname,p) = elt;
        info("  " + sname + "/" + p.uuid + " (" + p.name + ")");
    }
}

fn install_named(c: cargo, wd: str, name: str) {
    let mut ps = [];
    for_each_package(c, { |s, p|
        if p.name == name {
            vec::grow(ps, 1u, (s.name, p));
        }
    });
    if vec::len(ps) == 1u {
        let (_, p) = ps[0];
        install_package(c, wd, p);
        ret;
    } else if vec::len(ps) == 0u {
        cargo_suggestion(c, false, { ||
            error("can't find package: " + name);
        });
        ret;
    }
    error("found multiple packages:");
    for ps.each {|elt|
        let (sname,p) = elt;
        info("  " + sname + "/" + p.uuid + " (" + p.name + ")");
    }
}

fn install_uuid_specific(c: cargo, wd: str, src: str, uuid: str) {
    alt c.sources.find(src) {
      some(s) {
        let packages = copy s.packages;
        if vec::any(packages, { |p|
            if p.uuid == uuid {
                install_package(c, wd, p);
                true
            } else { false }
        }) { ret; }
      }
      _ { }
    }
    error("can't find package: " + src + "/" + uuid);
}

fn install_named_specific(c: cargo, wd: str, src: str, name: str) {
    alt c.sources.find(src) {
        some(s) {
          let packages = copy s.packages;
          if vec::any(packages, { |p|
                if p.name == name {
                    install_package(c, wd, p);
                    true
                } else { false }
            }) { ret; }
        }
        _ { }
    }
    error("can't find package: " + src + "/" + name);
}

fn cmd_uninstall(c: cargo) {
    if vec::len(c.opts.free) < 3u {
        cmd_usage();
        ret;
    }

    let lib = c.libdir;
    let bin = c.bindir;
    let target = c.opts.free[2u];

    // FIXME: needs stronger pattern matching
    // FIXME: needs to uninstall from a specified location in a cache instead
    // of looking for it (binaries can be uninstalled by name only)
    if is_uuid(target) {
        for os::list_dir(lib).each { |file|
            alt str::find_str(file, "-" + target + "-") {
                some(idx) {
                    let full = path::normalize(path::connect(lib, file));
                    if os::remove_file(full) {
                        info("uninstalled: '" + full + "'");
                    } else {
                        error("could not uninstall: '" + full + "'");
                    }
                    ret;
                }
                none { cont; }
            }
        }

        error("can't find package with uuid: " + target);
    } else {
        for os::list_dir(lib).each { |file|
            alt str::find_str(file, "lib" + target + "-") {
                some(idx) {
                    let full = path::normalize(path::connect(lib,
                               file));
                    if os::remove_file(full) {
                        info("uninstalled: '" + full + "'");
                    } else {
                        error("could not uninstall: '" + full + "'");
                    }
                    ret;
                }
                none { cont; }
            }
        }
        for os::list_dir(bin).each { |file|
            alt str::find_str(file, target) {
                some(idx) {
                    let full = path::normalize(path::connect(bin, file));
                    if os::remove_file(full) {
                        info("uninstalled: '" + full + "'");
                    } else {
                        error("could not uninstall: '" + full + "'");
                    }
                    ret;
                }
                none { cont; }
            }
        }

        error("can't find package with name: " + target);
    }
}

fn cmd_install(c: cargo) unsafe {
    // cargo install [pkg]
    if vec::len(c.opts.free) < 2u {
        cmd_usage();
        ret;
    }

    let wd_base = c.workdir + path::path_sep();
    let wd = alt tempfile::mkdtemp(wd_base, "") {
        some(_wd) { _wd }
        none { fail #fmt("needed temp dir: %s", wd_base); }
    };

    if vec::len(c.opts.free) == 2u {
        let cwd = os::getcwd();
        let status = run::run_program("cp", ["-R", cwd, wd]);

        if status != 0 {
            fail #fmt("could not copy directory: %s", cwd);
        }

        install_source(c, wd);
        ret;
    }

    let target = c.opts.free[2];

    if is_archive_url(target) {
        install_curl(c, wd, target);
    } else if is_git_url(target) {
        let ref = if c.opts.free.len() >= 4u {
            some(c.opts.free[3u])
        } else {
            none
        };
        install_git(c, wd, target, ref)
    } else if is_archive_uri(target) {
        install_file(c, wd, target);
        ret;
    } else {
        let mut ps = copy target;

        alt str::find_char(ps, '/') {
            option::some(idx) {
                let source = str::slice(ps, 0u, idx);
                ps = str::slice(ps, idx + 1u, str::len(ps));
                if is_uuid(ps) {
                    install_uuid_specific(c, wd, source, ps);
                } else {
                    install_named_specific(c, wd, source, ps);
                }
            }
            option::none {
                if is_uuid(ps) {
                    install_uuid(c, wd, ps);
                } else {
                    install_named(c, wd, ps);
                }
            }
        }
    }
}

fn sync_one(c: cargo, name: str, src: source) {
    let dir = path::connect(c.sourcedir, name);
    let pkgfile = path::connect(dir, "packages.json.new");
    let destpkgfile = path::connect(dir, "packages.json");
    let sigfile = path::connect(dir, "packages.json.sig");
    let keyfile = path::connect(dir, "key.gpg");
    let url = src.url;
    need_dir(dir);
    info(#fmt["fetching source %s...", name]);
    let p = run::program_output("curl", ["-f", "-s", "-o", pkgfile, url]);
    if p.status != 0 {
        warn(#fmt["fetch for source %s (url %s) failed", name, url]);
    } else {
        info(#fmt["fetched source: %s", name]);
    }
    alt src.sig {
        some(u) {
            let p = run::program_output("curl", ["-f", "-s", "-o", sigfile,
                                                 u]);
            if p.status != 0 {
                warn(#fmt["fetch for source %s (sig %s) failed", name, u]);
            }
        }
        _ { }
    }
    alt src.key {
        some(u) {
            let p = run::program_output("curl",  ["-f", "-s", "-o", keyfile,
                                                  u]);
            if p.status != 0 {
                warn(#fmt["fetch for source %s (key %s) failed", name, u]);
            }
            pgp::add(c.root, keyfile);
        }
        _ { }
    }
    alt (src.sig, src.key, src.keyfp) {
        (some(_), some(_), some(f)) {
            let r = pgp::verify(c.root, pkgfile, sigfile, f);
            if !r {
                warn(#fmt["signature verification failed for source %s",
                          name]);
            } else {
                info(#fmt["signature ok for source %s", name]);
            }
        }
        _ {
            info(#fmt["no signature for source %s", name]);
        }
    }
    copy_warn(pkgfile, destpkgfile);
}

fn cmd_sync(c: cargo) {
    if vec::len(c.opts.free) >= 3u {
        vec::iter_between(c.opts.free, 2u, vec::len(c.opts.free)) { |name|
            alt c.sources.find(name) {
                some(source) {
                    sync_one(c, name, source);
                }
                none {
                    error(#fmt("no such source: %s", name));
                }
            }
        }
    } else {
        cargo_suggestion(c, true, { || } );
        for c.sources.each_value { |v|
            sync_one(c, v.name, v);
        }
    }
}

fn cmd_init(c: cargo) {
    let srcurl = "http://www.rust-lang.org/cargo/sources.json";
    let sigurl = "http://www.rust-lang.org/cargo/sources.json.sig";

    let srcfile = path::connect(c.root, "sources.json.new");
    let sigfile = path::connect(c.root, "sources.json.sig");
    let destsrcfile = path::connect(c.root, "sources.json");

    let p = run::program_output("curl", ["-f", "-s", "-o", srcfile, srcurl]);
    if p.status != 0 {
        warn(#fmt["fetch of sources.json failed: %s", p.out]);
        ret;
    }

    let p = run::program_output("curl", ["-f", "-s", "-o", sigfile, sigurl]);
    if p.status != 0 {
        warn(#fmt["fetch of sources.json.sig failed: %s", p.out]);
        ret;
    }

    let r = pgp::verify(c.root, srcfile, sigfile, pgp::signing_key_fp());
    if !r {
        warn(#fmt["signature verification failed for '%s'", srcfile]);
    } else {
        info(#fmt["signature ok for '%s'", srcfile]);
    }
    copy_warn(srcfile, destsrcfile);

    info(#fmt["initialized .cargo in %s", c.root]);
}

fn print_pkg(s: source, p: package) {
    let mut m = s.name + "/" + p.name + " (" + p.uuid + ")";
    if vec::len(p.tags) > 0u {
        m = m + " [" + str::connect(p.tags, ", ") + "]";
    }
    info(m);
    if p.description != "" {
        print("   >> " + p.description + "\n")
    }
}

fn print_source(s: source) {
    info(s.name + " (" + s.url + ")");

    let pks = sort::merge_sort({ |a, b|
        a < b
    }, copy s.packages);
    let l = vec::len(pks);

    print(io::with_str_writer() { |writer|
        let mut list = "   >> ";

        vec::iteri(pks) { |i, pk|
            if str::len(list) > 78u {
                writer.write_line(list);
                list = "   >> ";
            }
            list += pk.name + (if l - 1u == i { "" } else { ", " });
        }

        writer.write_line(list);
    });
}

fn cmd_list(c: cargo) {
    if vec::len(c.opts.free) >= 3u {
        vec::iter_between(c.opts.free, 2u, vec::len(c.opts.free)) { |name|
            alt c.sources.find(name) {
                some(source) {
                    print_source(source);
                }
                none {
                    error(#fmt("no such source: %s", name));
                }
            }
        }
    } else {
        for c.sources.each_value { |v|
            print_source(v);
        }
    }
}

fn cmd_search(c: cargo) {
    if vec::len(c.opts.free) < 3u {
        cmd_usage();
        ret;
    }
    let mut n = 0;
    let name = c.opts.free[2];
    let tags = vec::slice(c.opts.free, 3u, vec::len(c.opts.free));
    for_each_package(c, { |s, p|
        if (str::contains(p.name, name) || name == "*") &&
            vec::all(tags, { |t| vec::contains(p.tags, t) }) {
            print_pkg(s, p);
            n += 1;
        }
    });
    info(#fmt["found %d packages", n]);
}

fn install_to_dir(srcfile: str, destdir: str) {
    let newfile = path::connect(destdir, path::basename(srcfile));

    let status = run::run_program("cp", [srcfile, newfile]);
    if status == 0 {
        info(#fmt["installed: '%s'", newfile]);
    } else {
        error(#fmt["could not install: '%s'", newfile]);
    }
}

fn copy_warn(srcfile: str, destfile: str) {
  if !os::copy_file(srcfile, destfile) {
      warn(#fmt["copying %s to %s failed", srcfile, destfile]);
  }
}

fn cmd_usage() {
    print("Usage: cargo <verb> [options] [args..]\n" +
          " e.g.: cargo [init | sync]\n" +
          " e.g.: cargo install [-g | -G] <package>

General:
    init                    Reinitialize cargo in ~/.cargo
    usage                   Display this message
    sync [sources..]        Sync all sources (or specific sources)

Querying:
    list [sources..]                        List sources and their packages
                                            or a single source
    search <name | '*'> [tags...]           Search packages

Packages:
    install [options]                       Install a package from source
                                            code in the current directory
    install [options] [source/]<name>       Install a package by name
    install [options] [source/]<uuid>       Install a package by uuid
    install [options] <url>                 Install a package via curl (HTTP,
                                            FTP, etc.) from an
                                            .tar[.gz|bz2|xz] file
    install [options] <url> [ref]           Install a package via git
    install [options] <file>                Install a package directly from an
                                            .tar[.gz|bz2|xz] file
    uninstall [options] <name>              Remove a package by (meta) name
    uninstall [options] <uuid>              Remove a package by (meta) uuid

Package installation options:
    --tests         Run crate tests before installing

Package [un]installation options:
    -g              Work at the user level (~/.cargo/bin/ instead of
                    locally in ./.cargo/bin/ by default)
    -G              Work at the system level (/usr/local/lib/cargo/bin/)

Other:
    -h, --help     Display this message
");
}

fn main(argv: [str]) {
    let o = build_cargo_options(argv);

    if vec::len(o.free) < 2u || o.help {
        cmd_usage();
        ret;
    }

    let mut c = configure(o);
    let mut sources = c.sources;
    let home = c.root;

    if !os::path_exists(path::connect(home, "sources.json")) {
        cmd_init(c);
        try_parse_sources(path::connect(home, "sources.json"), sources);
        try_parse_sources(path::connect(home, "local-sources.json"), sources);

        for sources.each_value { |v|
            sync_one(c, v.name, v);
        }

        // FIXME: shouldn't need to reconfigure
        c = configure(o);
    }

    alt o.free[1] {
        "init" { cmd_init(c); }
        "install" { cmd_install(c); }
        "uninstall" { cmd_uninstall(c); }
        "list" { cmd_list(c); }
        "search" { cmd_search(c); }
        "sync" { cmd_sync(c); }
        "usage" { cmd_usage(); }
        _ { cmd_usage(); }
    }
}
