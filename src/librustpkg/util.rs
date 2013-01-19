use core::*;
use send_map::linear::LinearMap;
use rustc::metadata::filesearch;
use semver::Version;
use std::{json, term, sort};

pub struct Package {
    id: ~str,
    vers: Version,
    bins: ~[~str],
    libs: ~[~str],
}

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

pub fn parse_name(id: ~str) -> result::Result<~str, ~str> {
    let parts = str::split_char(id, '.');

    for parts.each |&part| {
        for str::chars(part).each |&char| {
            if char::is_whitespace(char) {
                return result::Err(~"could not parse id: contains whitespace");
            } else if char::is_uppercase(char) {
                return result::Err(~"could not parse id: should be all lowercase");
            }
        }
    }

    result::Ok(parts.last())
}

pub fn parse_vers(vers: ~str) -> result::Result<Version, ~str> {
    match semver::parse(vers) {
        Some(vers) => result::Ok(vers),
        None => result::Err(~"could not parse version: invalid")
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

pub fn hash(data: ~str) -> ~str {
    let hasher = hash::default_state();

    hasher.write_str(data);
    hasher.result_str()
}

pub fn temp_change_dir<T>(dir: &Path, cb: fn() -> T) {
    let cwd = os::getcwd();

    os::change_dir(dir);
    cb();
    os::change_dir(&cwd);
}

pub fn touch(path: &Path) {
    match io::mk_file_writer(path, ~[io::Create]) {
        result::Ok(writer) => writer.write_line(~""),
        _ => {}
    }
}

pub fn remove_dir_r(path: &Path) {
    for os::walk_dir(path) |&file| {
        let mut cdir = file;

        loop {
            if os::path_is_dir(&cdir) {
                os::remove_dir(&cdir);
            } else {
                os::remove_file(&cdir);
            }

            cdir = cdir.dir_path();

            if cdir == *path { break; }
        }
    }

    os::remove_dir(path);
}

pub fn wait_for_lock(path: &Path) {
    if os::path_exists(path) {
        warn(fmt!("the database appears locked, please wait (or rm %s)",
                        path.to_str()));

        loop {
            if !os::path_exists(path) { break; }
        }
    }
}

fn _add_pkg(packages: ~[json::Json], pkg: &Package) -> ~[json::Json] {
    for packages.each |&package| {
        match package {
            json::Object(map) => {
                let mut has_id = false;

                match map.get(&~"id") {
                    json::String(str) => {
                        if pkg.id == str {
                            has_id = true;
                        }
                    }
                    _ => {}
                }

                match map.get(&~"vers") {
                    json::String(str) => {
                        if pkg.vers.to_str() == str {
                            return packages;
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    let mut map = ~LinearMap();

    map.insert(~"id", json::String(pkg.id));
    map.insert(~"vers", json::String(pkg.vers.to_str()));
    map.insert(~"bins", json::List(do pkg.bins.map |&bin| {
        json::String(bin)
    }));
    map.insert(~"libs", json::List(do pkg.libs.map |&lib| {
        json::String(lib)
    }));

    vec::append(packages, ~[json::Object(map)])
}

fn _rm_pkg(packages: ~[json::Json], pkg: &Package) -> ~[json::Json] {
    do packages.filter_map |&package| {
        match package {
            json::Object(map) => {
                let mut has_id = false;

                match map.get(&~"id") {
                    json::String(str) => {
                        if pkg.id == str {
                            has_id = true;
                        }
                    }
                    _ => {}
                }

                match map.get(&~"vers") {
                    json::String(str) => {
                        if pkg.vers.to_str() == str { None }
                        else { Some(package) } 
                    }
                    _ => { Some(package) }
                }
            }
            _ => { Some(package) }
        }
    }
}

pub fn load_pkgs() -> result::Result<~[json::Json], ~str> {
    let root = root();
    let db = root.push(~"db.json");
    let db_lock = root.push(~"db.json.lck");

    wait_for_lock(&db_lock);
    touch(&db_lock);

    let packages = if os::path_exists(&db) {
        match io::read_whole_file_str(&db) {
            result::Ok(str) => {
                match json::from_str(str) {
                    result::Ok(json) => {
                        match json {
                            json::List(list) => list,
                            _ => {
                                os::remove_file(&db_lock);

                                return result::Err(~"package db's json is not a list");
                            }
                        }
                    }
                    result::Err(err) => {
                        os::remove_file(&db_lock);

                        return result::Err(fmt!("failed to parse package db: %s", err.to_str()));
                    }
                }
            }
            result::Err(err) => {
                os::remove_file(&db_lock);

                return result::Err(fmt!("failed to read package db: %s", err));
            }
        }
    } else { ~[] };

    os::remove_file(&db_lock);

    result::Ok(packages)
}

pub fn get_pkg(id: ~str, vers: Option<~str>) -> result::Result<Package, ~str> {
    let name = match parse_name(id) {
        result::Ok(name) => name,
        result::Err(err) => return result::Err(err)
    };
    let packages = match load_pkgs() {
        result::Ok(packages) => packages,
        result::Err(err) => return result::Err(err)
    };
    let mut sel = None;
    let mut possibs = ~[];
    let mut err = None;

    for packages.each |&package| {
        match package {
            json::Object(map) => {
                let pid = match map.get(&~"id") {
                    json::String(str) => str,
                    _ => loop
                };
                let pname = match parse_name(pid) {
                    result::Ok(pname) => pname,
                    result::Err(perr) => {
                        err = Some(perr);

                        break;
                    }
                };
                let pvers = match map.get(&~"vers") {
                    json::String(str) => str,
                    _ => loop
                };
                if pid == id || pname == name {
                    let bins = match map.get(&~"bins") {
                        json::List(list) => {
                            do list.map |&bin| {
                                match bin {
                                    json::String(str) => str,
                                    _ => ~""
                                }
                            }
                        }
                        _ => ~[]
                    };
                    let libs = match map.get(&~"libs") {
                        json::List(list) => {
                            do list.map |&lib| {
                                match lib {
                                    json::String(str) => str,
                                    _ => ~""
                                }
                            }
                        }
                        _ => ~[]
                    };
                    let package = Package {
                        id: pid,
                        vers: match parse_vers(pvers) {
                            result::Ok(vers) => vers,
                            result::Err(verr) => {
                                err = Some(verr);

                                break;
                            }
                        },
                        bins: bins,
                        libs: libs
                    };

                    if !vers.is_none() && vers.get() == pvers {
                        sel = Some(package);
                    }
                    else {
                        possibs.push(package);
                    }
                }
            }
            _ => {}
        }
    }

    if !err.is_none() {
        return result::Err(err.get());
    }
    if !sel.is_none() {
        return result::Ok(sel.get());
    }
    if !vers.is_none() || possibs.len() < 1 {
        return result::Err(~"package not found");
    }

    result::Ok(sort::merge_sort(possibs, |v1, v2| {
        v1.vers <= v2.vers
    }).last())
}

pub fn add_pkg(pkg: &Package) -> bool {
    let root = root();
    let db = root.push(~"db.json");
    let db_lock = root.push(~"db.json.lck");
    let packages = match load_pkgs() {
        result::Ok(packages) => packages,
        result::Err(err) => {
            error(err);

            return false;
        }
    };

    wait_for_lock(&db_lock);
    touch(&db_lock);
    os::remove_file(&db);

    match io::mk_file_writer(&db, ~[io::Create]) {
        result::Ok(writer) => {
            writer.write_line(json::to_pretty_str(&json::List(_add_pkg(packages, pkg))));
        }
        result::Err(err) => {
            error(fmt!("failed to dump package db: %s", err));
            os::remove_file(&db_lock);

            return false;
        }
    }

    os::remove_file(&db_lock);

    true
}

pub fn remove_pkg(pkg: &Package) -> bool {
    let root = root();
    let db = root.push(~"db.json");
    let db_lock = root.push(~"db.json.lck");
    let packages = match load_pkgs() {
        result::Ok(packages) => packages,
        result::Err(err) => {
            error(err);

            return false;
        }
    };

    wait_for_lock(&db_lock);
    touch(&db_lock);
    os::remove_file(&db);

    match io::mk_file_writer(&db, ~[io::Create]) {
        result::Ok(writer) => {
            writer.write_line(json::to_pretty_str(&json::List(_rm_pkg(packages, pkg))));
        }
        result::Err(err) => {
            error(fmt!("failed to dump package db: %s", err));
            os::remove_file(&db_lock);

            return false;
        }
    }

    os::remove_file(&db_lock);

    true
}

#[cfg(windows)]
pub fn link_exe(_src: &Path, _dest: &Path) -> bool{
    /* FIXME: Investigate how to do this on win32
       Node wraps symlinks by having a .bat,
       but that won't work with minGW. */

    false
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
pub fn link_exe(src: &Path, dest: &Path) -> bool unsafe {
    do str::as_c_str(src.to_str()) |src_buf| {
        do str::as_c_str(dest.to_str()) |dest_buf| {
            libc::link(src_buf, dest_buf) == 0 as libc::c_int &&
            libc::chmod(dest_buf, 755) == 0 as libc::c_int
        }
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
fn test_parse_name() {
    assert parse_name(~"org.mozilla.servo").get() == ~"servo";
    assert parse_name(~"org. mozilla.servo 2131").is_err();
}
