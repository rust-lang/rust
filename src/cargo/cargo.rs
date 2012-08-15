// cargo.rs - Rust package manager

import syntax::{ast, codemap, parse, visit, attr};
import syntax::diagnostic::span_handler;
import codemap::span;
import rustc::metadata::filesearch::{get_cargo_root, get_cargo_root_nearest,
                                     get_cargo_sysroot, libdir};
import syntax::diagnostic;

import result::{ok, err};
import io::WriterUtil;
import std::{map, json, tempfile, term, sort, getopts};
import map::hashmap;
import to_str::to_str;
import getopts::{optflag, optopt, opt_present};

type package = {
    name: ~str,
    uuid: ~str,
    url: ~str,
    method: ~str,
    description: ~str,
    reference: option<~str>,
    tags: ~[~str],
    versions: ~[(~str, ~str)]
};

type local_package = {
    name: ~str,
    metaname: ~str,
    version: ~str,
    files: ~[~str]
};

type source = @{
    name: ~str,
    mut url: ~str,
    mut method: ~str,
    mut key: option<~str>,
    mut keyfp: option<~str>,
    mut packages: ~[mut package]
};

type cargo = {
    pgp: bool,
    root: ~str,
    installdir: ~str,
    bindir: ~str,
    libdir: ~str,
    workdir: ~str,
    sourcedir: ~str,
    sources: map::hashmap<~str, source>,
    mut current_install: ~str,
    dep_cache: map::hashmap<~str, bool>,
    opts: options
};

type crate = {
    name: ~str,
    vers: ~str,
    uuid: ~str,
    desc: option<~str>,
    sigs: option<~str>,
    crate_type: option<~str>,
    deps: ~[~str]
};

type options = {
    test: bool,
    mode: mode,
    free: ~[~str],
    help: bool,
};

enum mode { system_mode, user_mode, local_mode }

fn opts() -> ~[getopts::opt] {
    ~[optflag(~"g"), optflag(~"G"), optflag(~"test"),
     optflag(~"h"), optflag(~"help")]
}

fn info(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_green);
        out.write_str(~"info: ");
        term::reset(out);
        out.write_line(msg);
    } else { out.write_line(~"info: " + msg); }
}

fn warn(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_yellow);
        out.write_str(~"warning: ");
        term::reset(out);
        out.write_line(msg);
    }else { out.write_line(~"warning: " + msg); }
}

fn error(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_red);
        out.write_str(~"error: ");
        term::reset(out);
        out.write_line(msg);
    }
    else { out.write_line(~"error: " + msg); }
}

fn is_uuid(id: ~str) -> bool {
    let parts = str::split_str(id, ~"-");
    if vec::len(parts) == 5u {
        let mut correct = 0u;
        for vec::eachi(parts) |i, part| {
            fn is_hex_digit(ch: char) -> bool {
                ('0' <= ch && ch <= '9') ||
                ('a' <= ch && ch <= 'f') ||
                ('A' <= ch && ch <= 'F')
            }

            if !part.all(is_hex_digit) {
                return false;
            }

            match i {
                0u => {
                    if str::len(part) == 8u {
                        correct += 1u;
                    }
                }
                1u | 2u | 3u => {
                    if str::len(part) == 4u {
                        correct += 1u;
                    }
                }
                4u => {
                    if str::len(part) == 12u {
                        correct += 1u;
                    }
                }
                _ => { }
            }
        }
        if correct >= 5u {
            return true;
        }
    }
    return false;
}

#[test]
fn test_is_uuid() {
    assert is_uuid(~"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaafAF09");
    assert !is_uuid(~"aaaaaaaa-aaaa-aaaa-aaaaa-aaaaaaaaaaaa");
    assert !is_uuid(~"");
    assert !is_uuid(~"aaaaaaaa-aaa -aaaa-aaaa-aaaaaaaaaaaa");
    assert !is_uuid(~"aaaaaaaa-aaa!-aaaa-aaaa-aaaaaaaaaaaa");
    assert !is_uuid(~"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa-a");
    assert !is_uuid(~"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaป");
}

// FIXME (#2661): implement url/URL parsing so we don't have to resort
// to weak checks

fn has_archive_extension(p: ~str) -> bool {
    str::ends_with(p, ~".tar") ||
    str::ends_with(p, ~".tar.gz") ||
    str::ends_with(p, ~".tar.bz2") ||
    str::ends_with(p, ~".tar.Z") ||
    str::ends_with(p, ~".tar.lz") ||
    str::ends_with(p, ~".tar.xz") ||
    str::ends_with(p, ~".tgz") ||
    str::ends_with(p, ~".tbz") ||
    str::ends_with(p, ~".tbz2") ||
    str::ends_with(p, ~".tb2") ||
    str::ends_with(p, ~".taz") ||
    str::ends_with(p, ~".tlz") ||
    str::ends_with(p, ~".txz")
}

fn is_archive_path(u: ~str) -> bool {
    has_archive_extension(u) && os::path_exists(u)
}

fn is_archive_url(u: ~str) -> bool {
    // FIXME (#2661): this requires the protocol bit - if we had proper
    // url parsing, we wouldn't need it

    match str::find_str(u, ~"://") {
        option::some(i) => has_archive_extension(u),
        _ => false
    }
}

fn is_git_url(url: ~str) -> bool {
    if str::ends_with(url, ~"/") { str::ends_with(url, ~".git/") }
    else {
        str::starts_with(url, ~"git://") || str::ends_with(url, ~".git")
    }
}

fn assume_source_method(url: ~str) -> ~str {
    if is_git_url(url) {
        return ~"git";
    }
    if str::starts_with(url, ~"file://") || os::path_exists(url) {
        return ~"file";
    }

    ~"curl"
}

fn load_link(mis: ~[@ast::meta_item]) -> (option<~str>,
                                         option<~str>,
                                         option<~str>) {
    let mut name = none;
    let mut vers = none;
    let mut uuid = none;
    for mis.each |a| {
        match a.node {
            ast::meta_name_value(v, {node: ast::lit_str(s), span: _}) => {
                match *v {
                    ~"name" => name = some(*s),
                    ~"vers" => vers = some(*s),
                    ~"uuid" => uuid = some(*s),
                    _ => { }
                }
            }
            _ => fail ~"load_link: meta items must be name-values"
        }
    }
    (name, vers, uuid)
}

fn load_crate(filename: ~str) -> option<crate> {
    let sess = parse::new_parse_sess(none);
    let c = parse::parse_crate_from_crate_file(filename, ~[], sess);

    let mut name = none;
    let mut vers = none;
    let mut uuid = none;
    let mut desc = none;
    let mut sigs = none;
    let mut crate_type = none;

    for c.node.attrs.each |a| {
        match a.node.value.node {
            ast::meta_name_value(v, {node: ast::lit_str(s), span: _}) => {
                match *v {
                    ~"desc" => desc = some(*v),
                    ~"sigs" => sigs = some(*v),
                    ~"crate_type" => crate_type = some(*v),
                    _ => { }
                }
            }
            ast::meta_list(v, mis) => {
                if *v == ~"link" {
                    let (n, v, u) = load_link(mis);
                    name = n;
                    vers = v;
                    uuid = u;
                }
            }
            _ => {
                fail ~"crate attributes may not contain " +
                     ~"meta_words";
            }
        }
    }

    type env = @{
        mut deps: ~[~str]
    };

    fn goto_view_item(e: env, i: @ast::view_item) {
        match i.node {
            ast::view_item_use(ident, metas, id) => {
                let name_items =
                    attr::find_meta_items_by_name(metas, ~"name");
                let m = if name_items.is_empty() {
                    metas + ~[attr::mk_name_value_item_str(@~"name", *ident)]
                } else {
                    metas
                };
                let mut attr_name = ident;
                let mut attr_vers = ~"";
                let mut attr_from = ~"";

              for m.each |item| {
                    match attr::get_meta_item_value_str(item) {
                        some(value) => {
                            let name = attr::get_meta_item_name(item);

                            match *name {
                                ~"vers" => attr_vers = *value,
                                ~"from" => attr_from = *value,
                                _ => ()
                            }
                        }
                        none => ()
                    }
                }

                let query = if !str::is_empty(attr_from) {
                    attr_from
                } else {
                    if !str::is_empty(attr_vers) {
                        *attr_name + ~"@" + attr_vers
                    } else { *attr_name }
                };

                match *attr_name {
                    ~"std" | ~"core" => (),
                    _ => vec::push(e.deps, query)
                }
            }
            _ => ()
        }
    }
    fn goto_item(_e: env, _i: @ast::item) {
    }

    let e = @{
        mut deps: ~[]
    };
    let v = visit::mk_simple_visitor(@{
        visit_view_item: |a| goto_view_item(e, a),
        visit_item: |a| goto_item(e, a),
        with *visit::default_simple_visitor()
    });

    visit::visit_crate(*c, (), v);

    let deps = copy e.deps;

    match (name, vers, uuid) {
        (some(name0), some(vers0), some(uuid0)) => {
            some({
                name: name0,
                vers: vers0,
                uuid: uuid0,
                desc: desc,
                sigs: sigs,
                crate_type: crate_type,
                deps: deps })
        }
        _ => return none
    }
}

fn print(s: ~str) {
    io::stdout().write_line(s);
}

fn rest(s: ~str, start: uint) -> ~str {
    if (start >= str::len(s)) {
        ~""
    } else {
        str::slice(s, start, str::len(s))
    }
}

fn need_dir(s: ~str) {
    if os::path_is_dir(s) { return; }
    if !os::make_dir(s, 493_i32 /* oct: 755 */) {
        fail fmt!{"can't make_dir %s", s};
    }
}

fn valid_pkg_name(s: ~str) -> bool {
    fn is_valid_digit(c: char) -> bool {
        ('0' <= c && c <= '9') ||
        ('a' <= c && c <= 'z') ||
        ('A' <= c && c <= 'Z') ||
        c == '-' ||
        c == '_'
    }

    s.all(is_valid_digit)
}

fn parse_source(name: ~str, j: json::json) -> source {
    if !valid_pkg_name(name) {
        fail fmt!{"'%s' is an invalid source name", name};
    }

    match j {
        json::dict(j) => {
            let mut url = match j.find(~"url") {
                some(json::string(u)) => *u,
                _ => fail ~"needed 'url' field in source"
            };
            let method = match j.find(~"method") {
                some(json::string(u)) => *u,
                _ => assume_source_method(url)
            };
            let key = match j.find(~"key") {
                some(json::string(u)) => some(*u),
                _ => none
            };
            let keyfp = match j.find(~"keyfp") {
                some(json::string(u)) => some(*u),
                _ => none
            };
            if method == ~"file" {
                url = os::make_absolute(url);
            }
            return @{
                name: name,
                mut url: url,
                mut method: method,
                mut key: key,
                mut keyfp: keyfp,
                mut packages: ~[mut] };
        }
        _ => fail ~"needed dict value in source"
    };
}

fn try_parse_sources(filename: ~str, sources: map::hashmap<~str, source>) {
    if !os::path_exists(filename)  { return; }
    let c = io::read_whole_file_str(filename);
    match json::from_str(result::get(c)) {
        ok(json::dict(j)) => {
          for j.each |k, v| {
                sources.insert(k, parse_source(k, v));
                debug!{"source: %s", k};
            }
        }
        ok(_) => fail ~"malformed sources.json",
        err(e) => fail fmt!{"%s:%s", filename, e.to_str()}
    }
}

fn load_one_source_package(src: source, p: map::hashmap<~str, json::json>) {
    let name = match p.find(~"name") {
        some(json::string(n)) => {
            if !valid_pkg_name(*n) {
                warn(~"malformed source json: "
                     + src.name + ~", '" + *n + ~"'"+
                     ~" is an invalid name (alphanumeric, underscores and" +
                     ~" dashes only)");
                return;
            }
            *n
        }
        _ => {
            warn(~"malformed source json: " + src.name + ~" (missing name)");
            return;
        }
    };

    let uuid = match p.find(~"uuid") {
        some(json::string(n)) => {
            if !is_uuid(*n) {
                warn(~"malformed source json: "
                     + src.name + ~", '" + *n + ~"'"+
                     ~" is an invalid uuid");
                return;
            }
            *n
        }
        _ => {
            warn(~"malformed source json: " + src.name + ~" (missing uuid)");
            return;
        }
    };

    let url = match p.find(~"url") {
        some(json::string(n)) => *n,
        _ => {
            warn(~"malformed source json: " + src.name + ~" (missing url)");
            return;
        }
    };

    let method = match p.find(~"method") {
        some(json::string(n)) => *n,
        _ => {
            warn(~"malformed source json: "
                 + src.name + ~" (missing method)");
            return;
        }
    };

    let reference = match p.find(~"ref") {
        some(json::string(n)) => some(*n),
        _ => none
    };

    let mut tags = ~[];
    match p.find(~"tags") {
        some(json::list(js)) => {
          for (*js).each |j| {
                match j {
                    json::string(j) => vec::grow(tags, 1u, *j),
                    _ => ()
                }
            }
        }
        _ => ()
    }

    let description = match p.find(~"description") {
        some(json::string(n)) => *n,
        _ => {
            warn(~"malformed source json: " + src.name
                 + ~" (missing description)");
            return;
        }
    };

    let newpkg = {
        name: name,
        uuid: uuid,
        url: url,
        method: method,
        description: description,
        reference: reference,
        tags: tags,
        versions: ~[]
    };

    match vec::position(src.packages, |pkg| pkg.uuid == uuid) {
      some(idx) => {
        src.packages[idx] = newpkg;
        log(debug, ~"  updated package: " + src.name + ~"/" + name);
      }
      none => {
        vec::grow(src.packages, 1u, newpkg);
      }
    }

    log(debug, ~"  loaded package: " + src.name + ~"/" + name);
}

fn load_source_info(c: cargo, src: source) {
    let dir = path::connect(c.sourcedir, src.name);
    let srcfile = path::connect(dir, ~"source.json");
    if !os::path_exists(srcfile) { return; }
    let srcstr = io::read_whole_file_str(srcfile);
    match json::from_str(result::get(srcstr)) {
        ok(json::dict(s)) => {
            let o = parse_source(src.name, json::dict(s));

            src.key = o.key;
            src.keyfp = o.keyfp;
        }
        ok(_) => {
            warn(~"malformed source.json: " + src.name +
                 ~"(source info is not a dict)");
        }
        err(e) => {
            warn(fmt!{"%s:%s", src.name, e.to_str()});
        }
    };
}
fn load_source_packages(c: cargo, src: source) {
    log(debug, ~"loading source: " + src.name);
    let dir = path::connect(c.sourcedir, src.name);
    let pkgfile = path::connect(dir, ~"packages.json");
    if !os::path_exists(pkgfile) { return; }
    let pkgstr = io::read_whole_file_str(pkgfile);
    match json::from_str(result::get(pkgstr)) {
        ok(json::list(js)) => {
          for (*js).each |j| {
                match j {
                    json::dict(p) => {
                        load_one_source_package(src, p);
                    }
                    _ => {
                        warn(~"malformed source json: " + src.name +
                             ~" (non-dict pkg)");
                    }
                }
            }
        }
        ok(_) => {
            warn(~"malformed packages.json: " + src.name +
                 ~"(packages is not a list)");
        }
        err(e) => {
            warn(fmt!{"%s:%s", src.name, e.to_str()});
        }
    };
}

fn build_cargo_options(argv: ~[~str]) -> options {
    let matches = match getopts::getopts(argv, opts()) {
        result::ok(m) => m,
        result::err(f) => {
            fail fmt!{"%s", getopts::fail_str(f)};
        }
    };

    let test = opt_present(matches, ~"test");
    let G    = opt_present(matches, ~"G");
    let g    = opt_present(matches, ~"g");
    let help = opt_present(matches, ~"h") || opt_present(matches, ~"help");
    let len  = vec::len(matches.free);

    let is_install = len > 1u && matches.free[1] == ~"install";
    let is_uninstall = len > 1u && matches.free[1] == ~"uninstall";

    if G && g { fail ~"-G and -g both provided"; }

    if !is_install && !is_uninstall && (g || G) {
        fail ~"-g and -G are only valid for `install` and `uninstall|rm`";
    }

    let mode =
        if (!is_install && !is_uninstall) || g { user_mode }
        else if G { system_mode }
        else { local_mode };

    {test: test, mode: mode, free: matches.free, help: help}
}

fn configure(opts: options) -> cargo {
    let home = match get_cargo_root() {
        ok(home) => home,
        err(_err) => result::get(get_cargo_sysroot())
    };

    let get_cargo_dir = match opts.mode {
        system_mode => get_cargo_sysroot,
        user_mode => get_cargo_root,
        local_mode => get_cargo_root_nearest
    };

    let p = result::get(get_cargo_dir());

    let sources = map::str_hash();
    try_parse_sources(path::connect(home, ~"sources.json"), sources);
    try_parse_sources(path::connect(home, ~"local-sources.json"), sources);

    let dep_cache = map::str_hash();

    let mut c = {
        pgp: pgp::supported(),
        root: home,
        installdir: p,
        bindir: path::connect(p, ~"bin"),
        libdir: path::connect(p, ~"lib"),
        workdir: path::connect(p, ~"work"),
        sourcedir: path::connect(home, ~"sources"),
        sources: sources,
        mut current_install: ~"",
        dep_cache: dep_cache,
        opts: opts
    };

    need_dir(c.root);
    need_dir(c.installdir);
    need_dir(c.sourcedir);
    need_dir(c.workdir);
    need_dir(c.libdir);
    need_dir(c.bindir);

    for sources.each_key |k| {
        let mut s = sources.get(k);
        load_source_packages(c, s);
        sources.insert(k, s);
    }

    if c.pgp {
        pgp::init(c.root);
    } else {
        warn(~"command `gpg` was not found");
        warn(~"you have to install gpg from source " +
             ~" or package manager to get it to work correctly");
    }

    c
}

fn for_each_package(c: cargo, b: fn(source, package)) {
    for c.sources.each_value |v| {
        // FIXME (#2280): this temporary shouldn't be
        // necessary, but seems to be, for borrowing.
        let pks = copy v.packages;
        for vec::each(pks) |p| {
            b(v, p);
        }
    }
}

// Runs all programs in directory <buildpath>
fn run_programs(buildpath: ~str) {
    let newv = os::list_dir_path(buildpath);
    for newv.each |ct| {
        run::run_program(ct, ~[]);
    }
}

// Runs rustc in <path + subdir> with the given flags
// and returns <path + subdir>
fn run_in_buildpath(what: ~str, path: ~str, subdir: ~str, cf: ~str,
                    extra_flags: ~[~str]) -> option<~str> {
    let buildpath = path::connect(path, subdir);
    need_dir(buildpath);
    debug!{"%s: %s -> %s", what, cf, buildpath};
    let p = run::program_output(rustc_sysroot(),
                                ~[~"--out-dir", buildpath, cf] + extra_flags);
    if p.status != 0 {
        error(fmt!{"rustc failed: %d\n%s\n%s", p.status, p.err, p.out});
        return none;
    }
    some(buildpath)
}

fn test_one_crate(_c: cargo, path: ~str, cf: ~str) {
  let buildpath = match run_in_buildpath(~"testing", path, ~"/test", cf,
                                       ~[ ~"--test"]) {
      none => return,
      some(bp) => bp
  };
  run_programs(buildpath);
}

fn install_one_crate(c: cargo, path: ~str, cf: ~str) {
    let buildpath = match run_in_buildpath(~"installing", path,
                                         ~"/build", cf, ~[]) {
      none => return,
      some(bp) => bp
    };
    let newv = os::list_dir_path(buildpath);
    let exec_suffix = os::exe_suffix();
    for newv.each |ct| {
        if (exec_suffix != ~"" && str::ends_with(ct, exec_suffix)) ||
            (exec_suffix == ~"" && !str::starts_with(path::basename(ct),
                                                    ~"lib")) {
            debug!{"  bin: %s", ct};
            install_to_dir(ct, c.bindir);
            if c.opts.mode == system_mode {
                // FIXME (#2662): Put this file in PATH / symlink it so it can
                // be used as a generic executable
                // `cargo install -G rustray` and `rustray file.obj`
            }
        } else {
            debug!{"  lib: %s", ct};
            install_to_dir(ct, c.libdir);
        }
    }
}


fn rustc_sysroot() -> ~str {
    match os::self_exe_path() {
        some(path) => {
            let path = ~[path, ~"..", ~"bin", ~"rustc"];
            let rustc = path::normalize(path::connect_many(path));
            debug!{"  rustc: %s", rustc};
            rustc
        }
        none => ~"rustc"
    }
}

fn install_source(c: cargo, path: ~str) {
    debug!{"source: %s", path};
    os::change_dir(path);

    let mut cratefiles = ~[];
    for os::walk_dir(~".") |p| {
        if str::ends_with(p, ~".rc") {
            vec::push(cratefiles, p);
        }
    }

    if vec::is_empty(cratefiles) {
        fail ~"this doesn't look like a rust package (no .rc files)";
    }

    for cratefiles.each |cf| {
        match load_crate(cf) {
            none => again,
            some(crate) => {
              for crate.deps.each |query| {
                    // FIXME (#1356): handle cyclic dependencies
                    // (n.b. #1356 says "Cyclic dependency is an error
                    // condition")

                    let wd_base = c.workdir + path::path_sep();
                    let wd = match tempfile::mkdtemp(wd_base, ~"") {
                        some(wd) => wd,
                        none => fail fmt!{"needed temp dir: %s", wd_base}
                    };

                    install_query(c, wd, query);
                }

                os::change_dir(path);

                if c.opts.test {
                    test_one_crate(c, path, cf);
                }
                install_one_crate(c, path, cf);
            }
        }
    }
}

fn install_git(c: cargo, wd: ~str, url: ~str, reference: option<~str>) {
    run::program_output(~"git", ~[~"clone", url, wd]);
    if option::is_some(reference) {
        let r = option::get(reference);
        os::change_dir(wd);
        run::run_program(~"git", ~[~"checkout", r]);
    }

    install_source(c, wd);
}

fn install_curl(c: cargo, wd: ~str, url: ~str) {
    let tarpath = path::connect(wd, ~"pkg.tar");
    let p = run::program_output(~"curl", ~[~"-f", ~"-s", ~"-o",
                                         tarpath, url]);
    if p.status != 0 {
        fail fmt!{"fetch of %s failed: %s", url, p.err};
    }
    run::run_program(~"tar", ~[~"-x", ~"--strip-components=1",
                             ~"-C", wd, ~"-f", tarpath]);
    install_source(c, wd);
}

fn install_file(c: cargo, wd: ~str, path: ~str) {
    run::program_output(~"tar", ~[~"-x", ~"--strip-components=1",
                             ~"-C", wd, ~"-f", path]);
    install_source(c, wd);
}

fn install_package(c: cargo, src: ~str, wd: ~str, pkg: package) {
    let url = copy pkg.url;
    let method = match pkg.method {
        ~"git" => ~"git",
        ~"file" => ~"file",
        _ => ~"curl"
    };

    info(fmt!{"installing %s/%s via %s...", src, pkg.name, method});

    match method {
        ~"git" => install_git(c, wd, url, copy pkg.reference),
        ~"file" => install_file(c, wd, url),
        ~"curl" => install_curl(c, wd, copy url),
        _ => ()
    }
}

fn cargo_suggestion(c: cargo, fallback: fn())
{
    if c.sources.size() == 0u {
        error(~"no sources defined - you may wish to run " +
              ~"`cargo init`");
        return;
    }
    fallback();
}

fn install_uuid(c: cargo, wd: ~str, uuid: ~str) {
    let mut ps = ~[];
    for_each_package(c, |s, p| {
        if p.uuid == uuid {
            vec::grow(ps, 1u, (s.name, copy p));
        }
    });
    if vec::len(ps) == 1u {
        let (sname, p) = copy ps[0];
        install_package(c, sname, wd, p);
        return;
    } else if vec::len(ps) == 0u {
        cargo_suggestion(c, || {
            error(~"can't find package: " + uuid);
        });
        return;
    }
    error(~"found multiple packages:");
    for ps.each |elt| {
        let (sname,p) = copy elt;
        info(~"  " + sname + ~"/" + p.uuid + ~" (" + p.name + ~")");
    }
}

fn install_named(c: cargo, wd: ~str, name: ~str) {
    let mut ps = ~[];
    for_each_package(c, |s, p| {
        if p.name == name {
            vec::grow(ps, 1u, (s.name, copy p));
        }
    });
    if vec::len(ps) == 1u {
        let (sname, p) = copy ps[0];
        install_package(c, sname, wd, p);
        return;
    } else if vec::len(ps) == 0u {
        cargo_suggestion(c, || {
            error(~"can't find package: " + name);
        });
        return;
    }
    error(~"found multiple packages:");
    for ps.each |elt| {
        let (sname,p) = copy elt;
        info(~"  " + sname + ~"/" + p.uuid + ~" (" + p.name + ~")");
    }
}

fn install_uuid_specific(c: cargo, wd: ~str, src: ~str, uuid: ~str) {
    match c.sources.find(src) {
      some(s) => {
        let packages = copy s.packages;
        if vec::any(packages, |p| {
            if p.uuid == uuid {
                install_package(c, src, wd, p);
                true
            } else { false }
        }) { return; }
      }
      _ => ()
    }
    error(~"can't find package: " + src + ~"/" + uuid);
}

fn install_named_specific(c: cargo, wd: ~str, src: ~str, name: ~str) {
    match c.sources.find(src) {
        some(s) => {
          let packages = copy s.packages;
          if vec::any(packages, |p| {
                if p.name == name {
                    install_package(c, src, wd, p);
                    true
                } else { false }
            }) { return; }
        }
        _ => ()
    }
    error(~"can't find package: " + src + ~"/" + name);
}

fn cmd_uninstall(c: cargo) {
    if vec::len(c.opts.free) < 3u {
        cmd_usage();
        return;
    }

    let lib = c.libdir;
    let bin = c.bindir;
    let target = c.opts.free[2u];

    // FIXME (#2662): needs stronger pattern matching
    // FIXME (#2662): needs to uninstall from a specified location in a
    // cache instead of looking for it (binaries can be uninstalled by
    // name only)
    if is_uuid(target) {
        for os::list_dir(lib).each |file| {
            match str::find_str(file, ~"-" + target + ~"-") {
                some(idx) => {
                    let full = path::normalize(path::connect(lib, file));
                    if os::remove_file(full) {
                        info(~"uninstalled: '" + full + ~"'");
                    } else {
                        error(~"could not uninstall: '" + full + ~"'");
                    }
                    return;
                }
                none => again
            }
        }

        error(~"can't find package with uuid: " + target);
    } else {
        for os::list_dir(lib).each |file| {
            match str::find_str(file, ~"lib" + target + ~"-") {
                some(idx) => {
                    let full = path::normalize(path::connect(lib,
                               file));
                    if os::remove_file(full) {
                        info(~"uninstalled: '" + full + ~"'");
                    } else {
                        error(~"could not uninstall: '" + full + ~"'");
                    }
                    return;
                }
                none => again
            }
        }
        for os::list_dir(bin).each |file| {
            match str::find_str(file, target) {
                some(idx) => {
                    let full = path::normalize(path::connect(bin, file));
                    if os::remove_file(full) {
                        info(~"uninstalled: '" + full + ~"'");
                    } else {
                        error(~"could not uninstall: '" + full + ~"'");
                    }
                    return;
                }
                none => again
            }
        }

        error(~"can't find package with name: " + target);
    }
}

fn install_query(c: cargo, wd: ~str, target: ~str) {
    match c.dep_cache.find(target) {
        some(inst) => {
            if inst {
                return;
            }
        }
        none => ()
    }

    c.dep_cache.insert(target, true);

    if is_archive_path(target) {
        install_file(c, wd, target);
        return;
    } else if is_git_url(target) {
        let reference = if c.opts.free.len() >= 4u {
            some(c.opts.free[3u])
        } else {
            none
        };
        install_git(c, wd, target, reference);
    } else if !valid_pkg_name(target) && has_archive_extension(target) {
        install_curl(c, wd, target);
        return;
    } else {
        let mut ps = copy target;

        match str::find_char(ps, '/') {
            option::some(idx) => {
                let source = str::slice(ps, 0u, idx);
                ps = str::slice(ps, idx + 1u, str::len(ps));
                if is_uuid(ps) {
                    install_uuid_specific(c, wd, source, ps);
                } else {
                    install_named_specific(c, wd, source, ps);
                }
            }
            option::none => {
                if is_uuid(ps) {
                    install_uuid(c, wd, ps);
                } else {
                    install_named(c, wd, ps);
                }
            }
        }
    }

    // FIXME (#2662): This whole dep_cache and current_install thing is
    // a bit of a hack. It should be cleaned up in the future.

    if target == c.current_install {
        for c.dep_cache.each |k, _v| {
            c.dep_cache.remove(k);
        }

        c.current_install = ~"";
    }
}

fn cmd_install(c: cargo) unsafe {
    let wd_base = c.workdir + path::path_sep();
    let wd = match tempfile::mkdtemp(wd_base, ~"") {
        some(wd) => wd,
        none => fail fmt!{"needed temp dir: %s", wd_base}
    };

    if vec::len(c.opts.free) == 2u {
        let cwd = os::getcwd();
        let status = run::run_program(~"cp", ~[~"-R", cwd, wd]);

        if status != 0 {
            fail fmt!{"could not copy directory: %s", cwd};
        }

        install_source(c, wd);
        return;
    }

    sync(c);

    let query = c.opts.free[2];
    c.current_install = copy query;

    install_query(c, wd, copy query);
}

fn sync(c: cargo) {
    for c.sources.each_key |k| {
        let mut s = c.sources.get(k);
        sync_one(c, s);
        c.sources.insert(k, s);
    }
}

fn sync_one_file(c: cargo, dir: ~str, src: source) -> bool {
    let name = src.name;
    let srcfile = path::connect(dir, ~"source.json.new");
    let destsrcfile = path::connect(dir, ~"source.json");
    let pkgfile = path::connect(dir, ~"packages.json.new");
    let destpkgfile = path::connect(dir, ~"packages.json");
    let keyfile = path::connect(dir, ~"key.gpg");
    let srcsigfile = path::connect(dir, ~"source.json.sig");
    let sigfile = path::connect(dir, ~"packages.json.sig");
    let url = src.url;
    let mut has_src_file = false;

    if !os::copy_file(path::connect(url, ~"packages.json"), pkgfile) {
        error(fmt!{"fetch for source %s (url %s) failed", name, url});
        return false;
    }

    if os::copy_file(path::connect(url, ~"source.json"), srcfile) {
        has_src_file = false;
    }

    os::copy_file(path::connect(url, ~"source.json.sig"), srcsigfile);
    os::copy_file(path::connect(url, ~"packages.json.sig"), sigfile);

    match copy src.key {
        some(u) => {
            let p = run::program_output(~"curl",
                                        ~[~"-f", ~"-s", ~"-o", keyfile, u]);
            if p.status != 0 {
                error(fmt!{"fetch for source %s (key %s) failed", name, u});
                return false;
            }
            pgp::add(c.root, keyfile);
        }
        _ => ()
    }
    match (src.key, src.keyfp) {
        (some(_), some(f)) => {
            let r = pgp::verify(c.root, pkgfile, sigfile, f);

            if !r {
                error(fmt!{"signature verification failed for source %s",
                          name});
                return false;
            }

            if has_src_file {
                let e = pgp::verify(c.root, srcfile, srcsigfile, f);

                if !e {
                    error(fmt!{"signature verification failed for source %s",
                              name});
                    return false;
                }
            }
        }
        _ => ()
    }

    copy_warn(pkgfile, destpkgfile);

    if has_src_file {
        copy_warn(srcfile, destsrcfile);
    }

    os::remove_file(keyfile);
    os::remove_file(srcfile);
    os::remove_file(srcsigfile);
    os::remove_file(pkgfile);
    os::remove_file(sigfile);

    info(fmt!{"synced source: %s", name});

    return true;
}

fn sync_one_git(c: cargo, dir: ~str, src: source) -> bool {
    let name = src.name;
    let srcfile = path::connect(dir, ~"source.json");
    let pkgfile = path::connect(dir, ~"packages.json");
    let keyfile = path::connect(dir, ~"key.gpg");
    let srcsigfile = path::connect(dir, ~"source.json.sig");
    let sigfile = path::connect(dir, ~"packages.json.sig");
    let url = src.url;

    fn rollback(name: ~str, dir: ~str, insecure: bool) {
        fn msg(name: ~str, insecure: bool) {
            error(fmt!{"could not rollback source: %s", name});

            if insecure {
                warn(~"a past security check failed on source " +
                     name + ~" and rolling back the source failed -"
                     + ~" this source may be compromised");
            }
        }

        if !os::change_dir(dir) {
            msg(name, insecure);
        }
        else {
            let p = run::program_output(~"git", ~[~"reset", ~"--hard",
                                                ~"HEAD@{1}"]);

            if p.status != 0 {
                msg(name, insecure);
            }
        }
    }

    if !os::path_exists(path::connect(dir, ~".git")) {
        let p = run::program_output(~"git", ~[~"clone", url, dir]);

        if p.status != 0 {
            error(fmt!{"fetch for source %s (url %s) failed", name, url});
            return false;
        }
    }
    else {
        if !os::change_dir(dir) {
            error(fmt!{"fetch for source %s (url %s) failed", name, url});
            return false;
        }

        let p = run::program_output(~"git", ~[~"pull"]);

        if p.status != 0 {
            error(fmt!{"fetch for source %s (url %s) failed", name, url});
            return false;
        }
    }

    let has_src_file = os::path_exists(srcfile);

    match copy src.key {
        some(u) => {
            let p = run::program_output(~"curl",
                                        ~[~"-f", ~"-s", ~"-o", keyfile, u]);
            if p.status != 0 {
                error(fmt!{"fetch for source %s (key %s) failed", name, u});
                rollback(name, dir, false);
                return false;
            }
            pgp::add(c.root, keyfile);
        }
        _ => ()
    }
    match (src.key, src.keyfp) {
        (some(_), some(f)) => {
            let r = pgp::verify(c.root, pkgfile, sigfile, f);

            if !r {
                error(fmt!{"signature verification failed for source %s",
                          name});
                rollback(name, dir, false);
                return false;
            }

            if has_src_file {
                let e = pgp::verify(c.root, srcfile, srcsigfile, f);

                if !e {
                    error(fmt!{"signature verification failed for source %s",
                              name});
                    rollback(name, dir, false);
                    return false;
                }
            }
        }
        _ => ()
    }

    os::remove_file(keyfile);

    info(fmt!{"synced source: %s", name});

    return true;
}

fn sync_one_curl(c: cargo, dir: ~str, src: source) -> bool {
    let name = src.name;
    let srcfile = path::connect(dir, ~"source.json.new");
    let destsrcfile = path::connect(dir, ~"source.json");
    let pkgfile = path::connect(dir, ~"packages.json.new");
    let destpkgfile = path::connect(dir, ~"packages.json");
    let keyfile = path::connect(dir, ~"key.gpg");
    let srcsigfile = path::connect(dir, ~"source.json.sig");
    let sigfile = path::connect(dir, ~"packages.json.sig");
    let mut url = src.url;
    let smart = !str::ends_with(src.url, ~"packages.json");
    let mut has_src_file = false;

    if smart {
        url += ~"/packages.json";
    }

    let p = run::program_output(~"curl",
                                ~[~"-f", ~"-s", ~"-o", pkgfile, url]);

    if p.status != 0 {
        error(fmt!{"fetch for source %s (url %s) failed", name, url});
        return false;
    }
    if smart {
        url = src.url + ~"/source.json";
        let p =
            run::program_output(~"curl",
                                ~[~"-f", ~"-s", ~"-o", srcfile, url]);

        if p.status == 0 {
            has_src_file = true;
        }
    }

    match copy src.key {
        some(u) => {
            let p = run::program_output(~"curl",
                                        ~[~"-f", ~"-s", ~"-o", keyfile, u]);
            if p.status != 0 {
                error(fmt!{"fetch for source %s (key %s) failed", name, u});
                return false;
            }
            pgp::add(c.root, keyfile);
        }
        _ => ()
    }
    match (src.key, src.keyfp) {
        (some(_), some(f)) => {
            if smart {
                url = src.url + ~"/packages.json.sig";
            }
            else {
                url = src.url + ~".sig";
            }

            let mut p = run::program_output(~"curl", ~[~"-f", ~"-s", ~"-o",
                        sigfile, url]);
            if p.status != 0 {
                error(fmt!{"fetch for source %s (sig %s) failed", name, url});
                return false;
            }

            let r = pgp::verify(c.root, pkgfile, sigfile, f);

            if !r {
                error(fmt!{"signature verification failed for source %s",
                          name});
                return false;
            }

            if smart && has_src_file {
                url = src.url + ~"/source.json.sig";

                p = run::program_output(~"curl",
                                        ~[~"-f", ~"-s", ~"-o",
                                          srcsigfile, url]);
                if p.status != 0 {
                    error(fmt!{"fetch for source %s (sig %s) failed",
                          name, url});
                    return false;
                }

                let e = pgp::verify(c.root, srcfile, srcsigfile, f);

                if !e {
                    error(~"signature verification failed for " +
                          ~"source " + name);
                    return false;
                }
            }
        }
        _ => ()
    }

    copy_warn(pkgfile, destpkgfile);

    if smart && has_src_file {
        copy_warn(srcfile, destsrcfile);
    }

    os::remove_file(keyfile);
    os::remove_file(srcfile);
    os::remove_file(srcsigfile);
    os::remove_file(pkgfile);
    os::remove_file(sigfile);

    info(fmt!{"synced source: %s", name});

    return true;
}

fn sync_one(c: cargo, src: source) {
    let name = src.name;
    let dir = path::connect(c.sourcedir, name);

    info(fmt!{"syncing source: %s...", name});

    need_dir(dir);

    let result = match src.method {
        ~"git" => sync_one_git(c, dir, src),
        ~"file" => sync_one_file(c, dir, src),
        _ => sync_one_curl(c, dir, src)
    };

    if result {
        load_source_info(c, src);
        load_source_packages(c, src);
    }
}

fn cmd_init(c: cargo) {
    let srcurl = ~"http://www.rust-lang.org/cargo/sources.json";
    let sigurl = ~"http://www.rust-lang.org/cargo/sources.json.sig";

    let srcfile = path::connect(c.root, ~"sources.json.new");
    let sigfile = path::connect(c.root, ~"sources.json.sig");
    let destsrcfile = path::connect(c.root, ~"sources.json");

    let p =
        run::program_output(~"curl", ~[~"-f", ~"-s", ~"-o", srcfile, srcurl]);
    if p.status != 0 {
        error(fmt!{"fetch of sources.json failed: %s", p.out});
        return;
    }

    let p =
        run::program_output(~"curl", ~[~"-f", ~"-s", ~"-o", sigfile, sigurl]);
    if p.status != 0 {
        error(fmt!{"fetch of sources.json.sig failed: %s", p.out});
        return;
    }

    let r = pgp::verify(c.root, srcfile, sigfile, pgp::signing_key_fp());
    if !r {
        error(fmt!{"signature verification failed for '%s'", srcfile});
        return;
    }

    copy_warn(srcfile, destsrcfile);
    os::remove_file(srcfile);
    os::remove_file(sigfile);

    info(fmt!{"initialized .cargo in %s", c.root});
}

fn print_pkg(s: source, p: package) {
    let mut m = s.name + ~"/" + p.name + ~" (" + p.uuid + ~")";
    if vec::len(p.tags) > 0u {
        m = m + ~" [" + str::connect(p.tags, ~", ") + ~"]";
    }
    info(m);
    if p.description != ~"" {
        print(~"   >> " + p.description + ~"\n")
    }
}

fn print_source(s: source) {
    info(s.name + ~" (" + s.url + ~")");

    let pks = sort::merge_sort(sys::shape_lt, copy s.packages);
    let l = vec::len(pks);

    print(io::with_str_writer(|writer| {
        let mut list = ~"   >> ";

        do vec::iteri(pks) |i, pk| {
            if str::len(list) > 78u {
                writer.write_line(list);
                list = ~"   >> ";
            }
            list += pk.name + (if l - 1u == i { ~"" } else { ~", " });
        }

        writer.write_line(list);
    }));
}

fn cmd_list(c: cargo) {
    sync(c);

    if vec::len(c.opts.free) >= 3u {
        do vec::iter_between(c.opts.free, 2u, vec::len(c.opts.free)) |name| {
            if !valid_pkg_name(name) {
                error(fmt!{"'%s' is an invalid source name", name});
            } else {
                match c.sources.find(name) {
                    some(source) => {
                        print_source(source);
                    }
                    none => {
                        error(fmt!{"no such source: %s", name});
                    }
                }
            }
        }
    } else {
        for c.sources.each_value |v| {
            print_source(v);
        }
    }
}

fn cmd_search(c: cargo) {
    if vec::len(c.opts.free) < 3u {
        cmd_usage();
        return;
    }

    sync(c);

    let mut n = 0;
    let name = c.opts.free[2];
    let tags = vec::slice(c.opts.free, 3u, vec::len(c.opts.free));
    for_each_package(c, |s, p| {
        if (str::contains(p.name, name) || name == ~"*") &&
            vec::all(tags, |t| vec::contains(p.tags, t) ) {
            print_pkg(s, p);
            n += 1;
        }
    });
    info(fmt!{"found %d packages", n});
}

fn install_to_dir(srcfile: ~str, destdir: ~str) {
    let newfile = path::connect(destdir, path::basename(srcfile));

    let status = run::run_program(~"cp", ~[~"-r", srcfile, newfile]);
    if status == 0 {
        info(fmt!{"installed: '%s'", newfile});
    } else {
        error(fmt!{"could not install: '%s'", newfile});
    }
}

fn dump_cache(c: cargo) {
    need_dir(c.root);

    let out = path::connect(c.root, ~"cache.json");
    let _root = json::dict(map::str_hash());

    if os::path_exists(out) {
        copy_warn(out, path::connect(c.root, ~"cache.json.old"));
    }
}
fn dump_sources(c: cargo) {
    if c.sources.size() < 1u {
        return;
    }

    need_dir(c.root);

    let out = path::connect(c.root, ~"sources.json");

    if os::path_exists(out) {
        copy_warn(out, path::connect(c.root, ~"sources.json.old"));
    }

    match io::buffered_file_writer(out) {
        result::ok(writer) => {
            let hash = map::str_hash();
            let root = json::dict(hash);

          for c.sources.each |k, v| {
                let chash = map::str_hash();
                let child = json::dict(chash);

                chash.insert(~"url", json::string(@v.url));
                chash.insert(~"method", json::string(@v.method));

                match copy v.key {
                    some(key) => {
                        chash.insert(~"key", json::string(@key));
                    }
                    _ => ()
                }
                match copy v.keyfp {
                    some(keyfp) => {
                        chash.insert(~"keyfp", json::string(@keyfp));
                    }
                    _ => ()
                }

                hash.insert(k, child);
            }

            writer.write_str(json::to_str(root));
        }
        result::err(e) => {
            error(fmt!{"could not dump sources: %s", e});
        }
    }
}

fn copy_warn(srcfile: ~str, destfile: ~str) {
    if !os::copy_file(srcfile, destfile) {
        warn(fmt!{"copying %s to %s failed", srcfile, destfile});
    }
}

fn cmd_sources(c: cargo) {
    if vec::len(c.opts.free) < 3u {
        for c.sources.each_value |v| {
            info(fmt!{"%s (%s) via %s",
                      v.name, v.url, v.method});
        }
        return;
    }

    let action = c.opts.free[2u];

    match action {
        ~"clear" => {
          for c.sources.each_key |k| {
                c.sources.remove(k);
            }

            info(~"cleared sources");
        }
        ~"add" => {
            if vec::len(c.opts.free) < 5u {
                cmd_usage();
                return;
            }

            let name = c.opts.free[3u];
            let url = c.opts.free[4u];

            if !valid_pkg_name(name) {
                error(fmt!{"'%s' is an invalid source name", name});
                return;
            }

            match c.sources.find(name) {
                some(source) => {
                    error(fmt!{"source already exists: %s", name});
                }
                none => {
                    c.sources.insert(name, @{
                        name: name,
                        mut url: url,
                        mut method: assume_source_method(url),
                        mut key: none,
                        mut keyfp: none,
                        mut packages: ~[mut]
                    });
                    info(fmt!{"added source: %s", name});
                }
            }
        }
        ~"remove" => {
            if vec::len(c.opts.free) < 4u {
                cmd_usage();
                return;
            }

            let name = c.opts.free[3u];

            if !valid_pkg_name(name) {
                error(fmt!{"'%s' is an invalid source name", name});
                return;
            }

            match c.sources.find(name) {
                some(source) => {
                    c.sources.remove(name);
                    info(fmt!{"removed source: %s", name});
                }
                none => {
                    error(fmt!{"no such source: %s", name});
                }
            }
        }
        ~"set-url" => {
            if vec::len(c.opts.free) < 5u {
                cmd_usage();
                return;
            }

            let name = c.opts.free[3u];
            let url = c.opts.free[4u];

            if !valid_pkg_name(name) {
                error(fmt!{"'%s' is an invalid source name", name});
                return;
            }

            match c.sources.find(name) {
                some(source) => {
                    let old = copy source.url;
                    let method = assume_source_method(url);

                    source.url = url;
                    source.method = method;

                    c.sources.insert(name, source);

                    info(fmt!{"changed source url: '%s' to '%s'", old, url});
                }
                none => {
                    error(fmt!{"no such source: %s", name});
                }
            }
        }
        ~"set-method" => {
            if vec::len(c.opts.free) < 5u {
                cmd_usage();
                return;
            }

            let name = c.opts.free[3u];
            let method = c.opts.free[4u];

            if !valid_pkg_name(name) {
                error(fmt!{"'%s' is an invalid source name", name});
                return;
            }

            match c.sources.find(name) {
                some(source) => {
                    let old = copy source.method;

                    source.method = match method {
                        ~"git" => ~"git",
                        ~"file" => ~"file",
                        _ => ~"curl"
                    };

                    c.sources.insert(name, source);

                    info(fmt!{"changed source method: '%s' to '%s'", old,
                         method});
                }
                none => {
                    error(fmt!{"no such source: %s", name});
                }
            }
        }
        ~"rename" => {
            if vec::len(c.opts.free) < 5u {
                cmd_usage();
                return;
            }

            let name = c.opts.free[3u];
            let newn = c.opts.free[4u];

            if !valid_pkg_name(name) {
                error(fmt!{"'%s' is an invalid source name", name});
                return;
            }
            if !valid_pkg_name(newn) {
                error(fmt!{"'%s' is an invalid source name", newn});
                return;
            }

            match c.sources.find(name) {
                some(source) => {
                    c.sources.remove(name);
                    c.sources.insert(newn, source);
                    info(fmt!{"renamed source: %s to %s", name, newn});
                }
                none => {
                    error(fmt!{"no such source: %s", name});
                }
            }
        }
        _ => cmd_usage()
    }
}

fn cmd_usage() {
    print(~"Usage: cargo <cmd> [options] [args..]
e.g. cargo install <name>

Where <cmd> is one of:
    init, install, list, search, sources,
    uninstall, usage

Options:

    -h, --help                  Display this message
    <cmd> -h, <cmd> --help      Display help for <cmd>
");
}

fn cmd_usage_init() {
    print(~"cargo init

Re-initialize cargo in ~/.cargo. Clears all sources and then adds the
default sources from <www.rust-lang.org/sources.json>.");
}

fn cmd_usage_install() {
    print(~"cargo install
cargo install [source/]<name>[@version]
cargo install [source/]<uuid>[@version]
cargo install <git url> [ref]
cargo install <tarball url>
cargo install <tarball file>

Options:
    --test      Run crate tests before installing
    -g          Install to the user level (~/.cargo/bin/ instead of
                locally in ./.cargo/bin/ by default)
    -G          Install to the system level (/usr/local/lib/cargo/bin/)

Install a crate. If no arguments are supplied, it installs from
the current working directory. If a source is provided, only install
from that source, otherwise it installs from any source.");
}

fn cmd_usage_uninstall() {
    print(~"cargo uninstall [source/]<name>[@version]
cargo uninstall [source/]<uuid>[@version]
cargo uninstall <meta-name>[@version]
cargo uninstall <meta-uuid>[@version]

Options:
    -g          Remove from the user level (~/.cargo/bin/ instead of
                locally in ./.cargo/bin/ by default)
    -G          Remove from the system level (/usr/local/lib/cargo/bin/)

Remove a crate. If a source is provided, only remove
from that source, otherwise it removes from any source.
If a crate was installed directly (git, tarball, etc.), you can remove
it by metadata.");
}

fn cmd_usage_list() {
    print(~"cargo list [sources..]

If no arguments are provided, list all sources and their packages.
If source names are provided, list those sources and their packages.
");
}

fn cmd_usage_search() {
    print(~"cargo search <query | '*'> [tags..]

Search packages.");
}

fn cmd_usage_sources() {
    print(~"cargo sources
cargo sources add <name> <url>
cargo sources remove <name>
cargo sources rename <name> <new>
cargo sources set-url <name> <url>
cargo sources set-method <name> <method>

If no arguments are supplied, list all sources (but not their packages).

Commands:
    add             Add a source. The source method will be guessed
                    from the URL.
    remove          Remove a source.
    rename          Rename a source.
    set-url         Change the URL for a source.
    set-method      Change the method for a source.");
}

fn main(argv: ~[~str]) {
    let o = build_cargo_options(argv);

    if vec::len(o.free) < 2u {
        cmd_usage();
        return;
    }
    if o.help {
        match o.free[1] {
            ~"init" => cmd_usage_init(),
            ~"install" => cmd_usage_install(),
            ~"uninstall" => cmd_usage_uninstall(),
            ~"list" => cmd_usage_list(),
            ~"search" => cmd_usage_search(),
            ~"sources" => cmd_usage_sources(),
            _ => cmd_usage()
        }
        return;
    }
    if o.free[1] == ~"usage" {
        cmd_usage();
        return;
    }

    let mut c = configure(o);
    let home = c.root;
    let first_time = os::path_exists(path::connect(home, ~"sources.json"));

    if !first_time && o.free[1] != ~"init" {
        cmd_init(c);

        // FIXME (#2662): shouldn't need to reconfigure
        c = configure(o);
    }

    match o.free[1] {
        ~"init" => cmd_init(c),
        ~"install" => cmd_install(c),
        ~"uninstall" => cmd_uninstall(c),
        ~"list" => cmd_list(c),
        ~"search" => cmd_search(c),
        ~"sources" => cmd_sources(c),
        _ => cmd_usage()
    }

    dump_cache(c);
    dump_sources(c);
}
