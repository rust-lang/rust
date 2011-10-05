import std::os;
import std::fs;
import std::os_fs;
import std::vec;
import std::map;
import std::str;
import std::uint;
import metadata::cstore;
import driver::session;
import util::filesearch;

export get_rpath_flags, test;

#[cfg(target_os="linux")]
#[cfg(target_os="macos")]
fn get_rpath_flags(_sess: session::session, _out_filename: str) -> [str] {
    log "preparing the RPATH!";

    // FIXME
    /*
    let cwd = os::getcwd();
    let sysroot = sess.filesearch().sysroot();
    let output = out_filename;
    let libs = cstore::get_used_crate_files(sess.get_cstore());
    let target_triple = sess.get_opts().target_triple;
    let rpaths = get_rpaths(cwd, sysroot, output, libs, target_triple);
    */
    let rpaths = [];
    rpaths_to_flags(rpaths)
}

#[cfg(target_os="win32")]
fn get_rpath_flags(_sess: session::session, _out_filename: str) -> [str] {
    []
}

fn rpaths_to_flags(rpaths: [str]) -> [str] {
    vec::map({ |rpath| #fmt("-Wl,-rpath,%s",rpath)}, rpaths)
}

fn get_rpaths(cwd: fs::path, sysroot: fs::path,
              output: fs::path, libs: [fs::path],
              target_triple: str) -> [str] {
    log #fmt("cwd: %s", cwd);
    log #fmt("sysroot: %s", sysroot);
    log #fmt("output: %s", output);
    log #fmt("libs:");
    for libpath in libs {
        log #fmt("    %s", libpath);
    }
    log #fmt("target_triple: %s", target_triple);

    // Use relative paths to the libraries. Binaries can be moved
    // as long as they maintain the relative relationship to the
    // crates they depend on.
    let rpaths = get_rpaths_relative_to_output(cwd, output, libs);

    // Make backup absolute paths to the libraries. Binaries can
    // be moved as long as the crates they link against don't move.
    rpaths += get_absolute_rpaths(cwd, libs);

    // And a final backup rpath to the global library location.
    rpaths += [get_install_prefix_rpath(target_triple)];

    // Remove duplicates
    let rpaths = minimize_rpaths(rpaths);
    ret rpaths;
}

fn get_rpaths_relative_to_output(cwd: fs::path,
                                 output: fs::path,
                                 libs: [fs::path]) -> [str] {
    vec::map(bind get_rpath_relative_to_output(cwd, output, _), libs)
}

fn get_rpath_relative_to_output(cwd: fs::path,
                                output: fs::path,
                                lib: fs::path) -> str {
    "$ORIGIN" + fs::path_sep() + get_relative_to(
        get_absolute(cwd, output),
        get_absolute(cwd, lib))
}

// Find the relative path from one file to another
fn get_relative_to(abs1: fs::path, abs2: fs::path) -> fs::path {
    assert fs::path_is_absolute(abs1);
    assert fs::path_is_absolute(abs2);
    let normal1 = fs::normalize(abs1);
    let normal2 = fs::normalize(abs2);
    let split1 = str::split(normal1, os_fs::path_sep as u8);
    let split2 = str::split(normal2, os_fs::path_sep as u8);
    let len1 = vec::len(split1);
    let len2 = vec::len(split2);
    assert len1 > 0u;
    assert len2 > 0u;

    let max_common_path = uint::min(len1, len2) - 1u;
    let start_idx = 0u;
    while start_idx < max_common_path
        && split1[start_idx] == split2[start_idx] {
        start_idx += 1u;
    }

    let path = [];

    for each _ in uint::range(start_idx, len1 - 1u) {
        path += [".."];
    }

    path += vec::slice(split2, start_idx, len2 - 1u);

    check vec::is_not_empty(path);
    ret fs::connect_many(path);
}

fn get_absolute_rpaths(cwd: fs::path, libs: [fs::path]) -> [str] {
    vec::map(bind get_absolute_rpath(cwd, _), libs)
}

fn get_absolute_rpath(cwd: fs::path, lib: fs::path) -> str {
    get_absolute(cwd, lib)
}

fn get_absolute(cwd: fs::path, lib: fs::path) -> fs::path {
    if fs::path_is_absolute(lib) {
        lib
    } else {
        fs::connect(cwd, lib)
    }
}

fn get_install_prefix_rpath(target_triple: str) -> str {
    let install_prefix = #env("CFG_PREFIX");

    if install_prefix == "" {
        fail "rustc compiled without CFG_PREFIX environment variable";
    }

    let path = [install_prefix]
        + filesearch::relative_target_lib_path(target_triple);
    check vec::is_not_empty(path);
    fs::connect_many(path)
}

fn minimize_rpaths(rpaths: [str]) -> [str] {
    let set = map::new_str_hash::<()>();
    for rpath in rpaths { set.insert(rpath, ()); }
    let minimized = [];
    for each rpath in set.keys() { minimized += [rpath]; }
    ret minimized;
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
mod test {
    #[test]
    fn test_rpaths_to_flags() {
        let flags = rpaths_to_flags(["path1", "path2"]);
        assert flags == ["-Wl,-rpath,path1", "-Wl,-rpath,path2"];
    }

    #[test]
    fn test_get_absolute1() {
        let cwd = "/dir";
        let lib = "some/path/lib";
        let res = get_absolute(cwd, lib);
        assert res == "/dir/some/path/lib";
    }

    #[test]
    fn test_get_absolute2() {
        let cwd = "/dir";
        let lib = "/some/path/lib";
        let res = get_absolute(cwd, lib);
        assert res == "/some/path/lib";
    }

    #[test]
    fn test_prefix_rpath() {
        let res = get_install_prefix_rpath("triple");
        assert res == #env("CFG_PREFIX") + "/lib/rustc/triple/lib";
    }

    #[test]
    fn test_minimize() {
        let res = minimize_rpaths(["rpath1", "rpath2", "rpath1"]);
        assert res == ["rpath1", "rpath2"];
    }

    #[test]
    fn test_relative_to1() {
        let p1 = "/usr/bin/rustc";
        let p2 = "/usr/lib/mylib";
        let res = get_relative_to(p1, p2);
        assert res == "../lib";
    }

    #[test]
    fn test_relative_to2() {
        let p1 = "/usr/bin/rustc";
        let p2 = "/usr/bin/../lib/mylib";
        let res = get_relative_to(p1, p2);
        assert res == "../lib";
    }

    #[test]
    fn test_relative_to3() {
        let p1 = "/usr/bin/whatever/rustc";
        let p2 = "/usr/lib/whatever/mylib";
        let res = get_relative_to(p1, p2);
        assert res == "../../lib/whatever";
    }

    #[test]
    fn test_relative_to4() {
        let p1 = "/usr/bin/whatever/../rustc";
        let p2 = "/usr/lib/whatever/mylib";
        let res = get_relative_to(p1, p2);
        assert res == "../lib/whatever";
    }

    #[test]
    fn test_relative_to5() {
        let p1 = "/usr/bin/whatever/../rustc";
        let p2 = "/usr/lib/whatever/../mylib";
        let res = get_relative_to(p1, p2);
        assert res == "../lib";
    }

    #[test]
    fn test_relative_to6() {
        let p1 = "/1";
        let p2 = "/2/3";
        let res = get_relative_to(p1, p2);
        assert res == "2";
    }

    #[test]
    fn test_relative_to7() {
        let p1 = "/1/2";
        let p2 = "/3";
        let res = get_relative_to(p1, p2);
        assert res == "..";
    }

    #[test]
    fn test_rpath_relative() {
        let res = get_rpath_relative_to_output(
            "/usr", "bin/rustc", "lib/libstd.so");
        assert res == "$ORIGIN/../lib";
    }
}
