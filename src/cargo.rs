// cargo.rs - Rust package manager

use rustc;
use std;

import rustc::syntax::{ast, codemap, visit};
import rustc::syntax::parse::parser;

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

fn install_file(path: str) -> option::t<str> {
    let wd = tempfile::mkdtemp("/tmp/cargo-work-", "");
    ret wd;
}

fn cmd_install(argv: [str]) {
    // cargo install <pkg>
    if vec::len(argv) < 3u {
        cmd_usage();
        ret;
    }

    let wd = if str::starts_with(argv[2], "file:") {
        let path = rest(argv[2], 5u);
        install_file(path)
    } else {
        none
    };
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
        "usage" { cmd_usage(); }
        _ { cmd_usage(); }
    }
}

