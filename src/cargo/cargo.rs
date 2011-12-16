// cargo.rs - Rust package manager

import rustc::syntax::{ast, codemap, visit};
import rustc::syntax::parse::parser;

import std::fs;
import std::generic_os;
import std::io;
import std::json;
import option;
import option::{none, some};
import std::os;
import std::run;
import str;
import std::tempfile;
import vec;

type cargo = {
    root: str,
    bindir: str,
    libdir: str,
    workdir: str,
    fetchdir: str
};

type pkg = {
    name: str,
    vers: str,
    uuid: str,
    desc: option::t<str>,
    sigs: option::t<str>,
    crate_type: option::t<str>
};

fn load_link(mis: [@ast::meta_item]) -> (option::t<str>,
                                         option::t<str>,
                                         option::t<str>) {
    let name = none;
    let vers = none;
    let uuid = none;
    for a: @ast::meta_item in mis {
        alt a.node {
            ast::meta_name_value(v, {node: ast::lit_str(s), span: _}) {
                alt v {
                    "name" { name = some(s); }
                    "vers" { vers = some(s); }
                    "uuid" { uuid = some(s); }
                    _ { }
                }
            }
        }
    }
    (name, vers, uuid)
}

fn load_pkg(filename: str) -> option::t<pkg> {
    let sess = @{cm: codemap::new_codemap(), mutable next_id: 0};
    let c = parser::parse_crate_from_crate_file(filename, [], sess);

    let name = none;
    let vers = none;
    let uuid = none;
    let desc = none;
    let sigs = none;
    let crate_type = none;

    for a in c.node.attrs {
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
    if (start >= str::char_len(s)) {
        ""
    } else {
        str::char_slice(s, start, str::char_len(s))
    }
}

fn need_dir(s: str) {
    if fs::file_is_dir(s) { ret; }
    if !fs::make_dir(s, 0x1c0i32) {
        fail #fmt["can't make_dir %s", s];
    }
}

fn configure() -> cargo {
    let p = alt generic_os::getenv("CARGO_ROOT") {
        some(_p) { _p }
        none. {
            alt generic_os::getenv("HOME") {
                some(_q) { fs::connect(_q, ".cargo") }
                none. { fail "no CARGO_ROOT or HOME"; }
            }
        }
    };

    log #fmt["p: %s", p];

    let c = {
        root: p,
        bindir: fs::connect(p, "bin"),
        libdir: fs::connect(p, "lib"),
        workdir: fs::connect(p, "work"),
        fetchdir: fs::connect(p, "fetch")
    };

    need_dir(c.root);
    need_dir(c.fetchdir);
    need_dir(c.workdir);
    need_dir(c.libdir);
    need_dir(c.bindir);

    c
}

fn install_one_crate(c: cargo, _path: str, cf: str, _p: pkg) {
    let name = fs::basename(cf);
    let ri = str::index(name, '.' as u8);
    if ri != -1 {
        name = str::slice(name, 0u, ri as uint);
    }
    log #fmt["Installing: %s", name];
    let old = vec::map(fs::list_dir("."),
                       {|x| str::slice(x, 2u, str::byte_len(x))});
    run::run_program("rustc", [name + ".rc"]);
    let new = vec::map(fs::list_dir("."),
                       {|x| str::slice(x, 2u, str::byte_len(x))});
    let created =
        vec::filter::<str>(new, { |n| !vec::member::<str>(n, old) });
    let exec_suffix = os::exec_suffix();
    for ct: str in created {
        if (exec_suffix != "" && str::ends_with(ct, exec_suffix)) ||
            (exec_suffix == "" && !str::starts_with(ct, "lib")) {
            log #fmt["  bin: %s", ct];
            // FIXME: need libstd fs::copy or something
            run::run_program("cp", [ct, c.bindir]);
        } else {
            log #fmt["  lib: %s", ct];
            run::run_program("cp", [ct, c.libdir]);
        }
    }
}

fn install_source(c: cargo, path: str) {
    log #fmt["source: %s", path];
    fs::change_dir(path);
    let contents = fs::list_dir(".");

    log #fmt["contents: %s", str::connect(contents, ", ")];

    let cratefiles =
        vec::filter::<str>(contents, { |n| str::ends_with(n, ".rc") });

    if vec::is_empty(cratefiles) {
        fail "This doesn't look like a rust package (no .rc files).";
    }

    for cf: str in cratefiles {
        let p = load_pkg(cf);
        alt p {
            none. { cont; }
            some(_p) {
                install_one_crate(c, path, cf, _p);
            }
        }
    }
}

fn install_git(c: cargo, wd: str, _path: str) {
    run::run_program("git", ["clone", _path, wd]);
    install_source(c, wd);
}

fn install_file(c: cargo, wd: str, _path: str) {
    run::run_program("tar", ["-x", "--strip-components=1",
                             "-C", wd, "-f", _path]);
    install_source(c, wd);
}

fn install_resolved(c: cargo, wd: str, key: str) {
    fs::remove_dir(wd);
    let u = "https://rust-package-index.appspot.com/pkg/" + key;
    let p = run::program_output("curl", [u]);
    if p.status != 0 {
        fail #fmt["Fetch of %s failed: %s", u, p.err];
    }
    let j = json::from_str(p.out);
    alt j {
        some (json::dict(_j)) {
            alt _j.find("install") {
                some (json::string(g)) {
                    log #fmt["Resolved: %s -> %s", key, g];
                    cmd_install(c, ["cargo", "install", g]);
                }
                _ { fail #fmt["Bogus install: '%s'", p.out]; }
            }
        }
        _ { fail #fmt["Bad json: '%s'", p.out]; }
    }
}

fn install_uuid(c: cargo, wd: str, uuid: str) {
    install_resolved(c, wd, "by-uuid/" + uuid);
}

fn install_named(c: cargo, wd: str, name: str) {
    install_resolved(c, wd, "by-name/" + name);
}

fn cmd_install(c: cargo, argv: [str]) {
    // cargo install <pkg>
    if vec::len(argv) < 3u {
        cmd_usage();
        ret;
    }

    let wd = alt tempfile::mkdtemp(c.workdir + fs::path_sep(), "") {
        some(_wd) { _wd }
        none. { fail "needed temp dir"; }
    };

    if str::starts_with(argv[2], "git:") {
        install_git(c, wd, argv[2]);
    } else if str::starts_with(argv[2], "github:") {
        let path = rest(argv[2], 7u);
        install_git(c, wd, "git://github.com/" + path);
    } else if str::starts_with(argv[2], "file:") {
        let path = rest(argv[2], 5u);
        install_file(c, wd, path);
    } else if str::starts_with(argv[2], "uuid:") {
        let uuid = rest(argv[2], 5u);
        install_uuid(c, wd, uuid);
    } else {
        install_named(c, wd, argv[2]);
    }
}

fn cmd_usage() {
    print("Usage: cargo <verb> [args...]");
}

fn main(argv: [str]) {
    if vec::len(argv) < 2u {
        cmd_usage();
        ret;
    }
    let c = configure();
    alt argv[1] {
        "install" { cmd_install(c, argv); }
        "usage" { cmd_usage(); }
        _ { cmd_usage(); }
    }
}
