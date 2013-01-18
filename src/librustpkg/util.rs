use core::*;
use rustc::metadata::filesearch;
use semver::Version;
use std::term;

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

pub fn need_dir(s: &Path) {
    if !os::path_is_dir(s) && !os::make_dir(s, 493_i32) {
        fail fmt!("can't create dir: %s", s.to_str());
    }
}

pub fn note(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_green);
        out.write_str(~"note: ");
        term::reset(out);
        out.write_line(msg);
    } else { out.write_line(~"note: " + msg); }
}

pub fn warn(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_yellow);
        out.write_str(~"warning: ");
        term::reset(out);
        out.write_line(msg);
    }else { out.write_line(~"warning: " + msg); }
}

pub fn error(msg: ~str) {
    let out = io::stdout();

    if term::color_supported() {
        term::fg(out, term::color_red);
        out.write_str(~"error: ");
        term::reset(out);
        out.write_line(msg);
    }
    else { out.write_line(~"error: " + msg); }
}

pub fn temp_change_dir<T>(dir: &Path, cb: fn() -> T) {
    let cwd = os::getcwd();

    os::change_dir(dir);
    cb();
    os::change_dir(&cwd);
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
