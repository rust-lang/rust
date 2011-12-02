// cargo.rs - Rust package manager

import rustc::syntax::{ast, codemap, visit};
import rustc::syntax::parse::parser;

import std::fs;
import std::io;
import std::option;
import std::option::{none, some};
import std::run;
import std::str;
import std::tempfile;
import std::vec;

type pkg = {
    name: str,
    vers: str,
    uuid: str,
    desc: option::t<str>,
    sigs: option::t<str>
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

    for a in c.node.attrs {
        alt a.node.value.node {
            ast::meta_name_value(v, {node: ast::lit_str(s), span: _}) {
                alt v {
                    "desc" { desc = some(v); }
                    "sigs" { sigs = some(v); }
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
                sigs: sigs})
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

fn install_source(path: str) {
    log #fmt["source: %s", path];
    fs::change_dir(path);
    let contents = fs::list_dir(".");

    log #fmt["contents: %s", str::connect(contents, ", ")];

    let cratefile = vec::find::<str>({ |n| str::ends_with(n, ".rc") }, contents);

    // First, try a configure script:
    if vec::member("./configure", contents) {
        run::run_program("./configure", []);
    }

    // Makefile?
    if vec::member("./Makefile", contents) {
        run::run_program("make", ["RUSTC=rustc"]);
    } else if option::is_some::<str>(cratefile) {
        run::run_program("rustc", [option::get(cratefile)]);
    }
}

fn install_file(_path: str) {
    let wd = tempfile::mkdtemp("/tmp/cargo-work-", "");
    alt wd {
        some(p) {
            run::run_program("tar", ["-x", "--strip-components=1",
                                     "-C", p, "-f", _path]);
            install_source(p);
        }
        _ { }
    }
}

fn cmd_install(argv: [str]) {
    // cargo install <pkg>
    if vec::len(argv) < 3u {
        cmd_usage();
        ret;
    }

    if str::starts_with(argv[2], "file:") {
        let path = rest(argv[2], 5u);
        install_file(path);
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
    alt argv[1] {
        "install" { cmd_install(argv); }
        "usage" { cmd_usage(); }
        _ { cmd_usage(); }
    }
}
