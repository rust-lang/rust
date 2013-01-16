use core::*;
use rustc::metadata::filesearch;
use semver::Version;
use std::net::url;

pub fn root() -> Path {
    match filesearch::get_rustpkg_root() {
        result::Ok(path) => path,
        result::Err(err) => fail err
    }
}

pub fn is_cmd(cmd: ~str) -> bool {
    let cmds = &[~"build", ~"clean", ~"install", ~"prefer", ~"test",
                 ~"uninstall", ~"unprefer"];

    vec::contains(cmds, &cmd)
}

pub fn parse_id(id: ~str) -> ~str {
    let parts = str::split_char(id, '.');

    for parts.each |&part| {
        for str::chars(part).each |&char| {
            if char::is_whitespace(char) {
                fail ~"could not parse id: contains whitespace";
            } else if char::is_uppercase(char) {
                fail ~"could not parse id: should be all lowercase";
            }
        }
    }

    parts.last()
}

pub fn parse_vers(vers: ~str) -> Version {
    match semver::parse(vers) {
        Some(vers) => vers,
        None => fail ~"could not parse version: invalid"
    }
}

#[test]
fn test_is_cmd() {
    assert is_cmd(~"build");
    assert is_cmd(~"clean");
    assert is_cmd(~"install");
    assert is_cmd(~"prefer");
    assert is_cmd(~"test");
    assert is_cmd(~"uninstall");
    assert is_cmd(~"unprefer");
}

#[test]
fn test_parse_id() {
    assert parse_id(~"org.mozilla.servo").get() == ~"servo";
    assert parse_id(~"org. mozilla.servo 2131").is_err();
}
