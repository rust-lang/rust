// cargo.rs - Rust package manager

import rustc::syntax::{ast, codemap, visit};
import rustc::syntax::parse::parser;

import std::fs;
import std::generic_os;
import std::io;
import std::option;
import std::option::{none, some};
import std::os;
import std::run;
import std::str;
import std::tempfile;
import std::vec;

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

fn setup_dirs() -> str {
    let p = alt generic_os::getenv("CARGO_ROOT") {
        some(_p) { _p }
        none. {
            alt generic_os::getenv("HOME") {
                some(_q) { fs::connect(_q, "/.cargo") }
                none. { fail "no CARGO_ROOT or HOME"; }
            }
        }
    };

    log #fmt["p: %s", p];

    need_dir(p);
    need_dir(fs::connect(p, "fetch"));
    need_dir(fs::connect(p, "work"));
    need_dir(fs::connect(p, "lib"));
    need_dir(fs::connect(p, "bin"));

    p
}

fn install_one_crate(cargo_root: str, path: str, cf: str, p: pkg) {
    let bindir = fs::connect(cargo_root, "bin");
    let libdir = fs::connect(cargo_root, "lib");
    let name = fs::basename(cf);
    let ri = str::index(name, '.' as u8);
    if ri != -1 {
        name = str::slice(name, 0u, ri as uint);
    }
    log #fmt["Installing: %s", name];
    let old = fs::list_dir(".");
    run::run_program("rustc", [cf]);
    let new = fs::list_dir(".");
    let created = vec::filter::<str>({ |n| !vec::member::<str>(n, old) }, new);
    for c: str in created {
        if str::ends_with(c, os::exec_suffix()) {
            log #fmt["  bin: %s", c];
            // FIXME: need libstd fs::copy or something
            run::run_program("cp", [c, fs::connect(bindir, c)]);
        } else {
            log #fmt["  lib: %s", c];
            run::run_program("cp", [c, fs::connect(libdir, c)]);
        }
    }
}

fn install_source(cargo_root: str, path: str) {
    log #fmt["source: %s", path];
    fs::change_dir(path);
    let contents = fs::list_dir(".");

    log #fmt["contents: %s", str::connect(contents, ", ")];

    let cratefiles = vec::filter::<str>({ |n| str::ends_with(n, ".rc") }, contents);

    if vec::is_empty(cratefiles) {
        fail "This doesn't look like a rust package (no .rc files).";
    }

    for cf: str in cratefiles {
        let p = load_pkg(cf);
        alt p {
            none. { cont; }
            some(_p) {
                install_one_crate(cargo_root, path, cf, _p);
            }
        }
    }
}

fn install_file(cargo_root: str, _path: str) {
    let wd = tempfile::mkdtemp(cargo_root + "/work/", "");
    alt wd {
        some(p) {
            run::run_program("tar", ["-x", "--strip-components=1",
                                     "-C", p, "-f", _path]);
            install_source(cargo_root, p);
        }
        _ { fail "needed temp dir"; }
    }
}

fn cmd_install(cargo_root: str, argv: [str]) {
    // cargo install <pkg>
    if vec::len(argv) < 3u {
        cmd_usage();
        ret;
    }

    if str::starts_with(argv[2], "file:") {
        let path = rest(argv[2], 5u);
        install_file(cargo_root, path);
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
    let cargo_root = setup_dirs();
    alt argv[1] {
        "install" { cmd_install(cargo_root, argv); }
        "usage" { cmd_usage(); }
        _ { cmd_usage(); }
    }
}
