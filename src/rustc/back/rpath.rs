import std::map;
import std::map::hashmap;
import metadata::cstore;
import driver::session;
import metadata::filesearch;

export get_rpath_flags;

pure fn not_win32(os: session::os) -> bool {
  alt os {
      session::os_win32 { false }
      _ { true }
  }
}

fn get_rpath_flags(sess: session::session, out_filename: str) -> [str]/~ {
    let os = sess.targ_cfg.os;

    // No rpath on windows
    if os == session::os_win32 {
        ret []/~;
    }

    #debug("preparing the RPATH!");

    let cwd = os::getcwd();
    let sysroot = sess.filesearch.sysroot();
    let output = out_filename;
    let libs = cstore::get_used_crate_files(sess.cstore);
    // We don't currently rpath native libraries, but we know
    // where rustrt is and we know every rust program needs it
    let libs = libs + [get_sysroot_absolute_rt_lib(sess)]/~;

    let target_triple = sess.opts.target_triple;
    let rpaths = get_rpaths(os, cwd, sysroot, output, libs, target_triple);
    rpaths_to_flags(rpaths)
}

fn get_sysroot_absolute_rt_lib(sess: session::session) -> path::path {
    let path = [sess.filesearch.sysroot()]/~
        + filesearch::relative_target_lib_path(
            sess.opts.target_triple)
        + [os::dll_filename("rustrt")]/~;
    path::connect_many(path)
}

fn rpaths_to_flags(rpaths: [str]/~) -> [str]/~ {
    vec::map(rpaths, { |rpath| #fmt("-Wl,-rpath,%s",rpath)})
}

fn get_rpaths(os: session::os, cwd: path::path, sysroot: path::path,
              output: path::path, libs: [path::path]/~,
              target_triple: str) -> [str]/~ {
    #debug("cwd: %s", cwd);
    #debug("sysroot: %s", sysroot);
    #debug("output: %s", output);
    #debug("libs:");
    for libs.each {|libpath|
        #debug("    %s", libpath);
    }
    #debug("target_triple: %s", target_triple);

    // Use relative paths to the libraries. Binaries can be moved
    // as long as they maintain the relative relationship to the
    // crates they depend on.
    let rel_rpaths = get_rpaths_relative_to_output(os, cwd, output, libs);

    // Make backup absolute paths to the libraries. Binaries can
    // be moved as long as the crates they link against don't move.
    let abs_rpaths = get_absolute_rpaths(cwd, libs);

    // And a final backup rpath to the global library location.
    let fallback_rpaths = [get_install_prefix_rpath(cwd, target_triple)]/~;

    fn log_rpaths(desc: str, rpaths: [str]/~) {
        #debug("%s rpaths:", desc);
        for rpaths.each {|rpath|
            #debug("    %s", rpath);
        }
    }

    log_rpaths("relative", rel_rpaths);
    log_rpaths("absolute", abs_rpaths);
    log_rpaths("fallback", fallback_rpaths);

    let rpaths = rel_rpaths + abs_rpaths + fallback_rpaths;

    // Remove duplicates
    let rpaths = minimize_rpaths(rpaths);
    ret rpaths;
}

fn get_rpaths_relative_to_output(os: session::os,
                                 cwd: path::path,
                                 output: path::path,
                                 libs: [path::path]/~) -> [str]/~ {
    vec::map(libs, {|a|
        check not_win32(os);
        get_rpath_relative_to_output(os, cwd, output, a)
    })
}

fn get_rpath_relative_to_output(os: session::os,
                                cwd: path::path,
                                output: path::path,
                                &&lib: path::path) : not_win32(os) -> str {
    // Mac doesn't appear to support $ORIGIN
    let prefix = alt os {
        session::os_linux { "$ORIGIN" + path::path_sep() }
        session::os_freebsd { "$ORIGIN" + path::path_sep() }
        session::os_macos { "@executable_path" + path::path_sep() }
        session::os_win32 { core::unreachable(); }
    };

    prefix + get_relative_to(
        get_absolute(cwd, output),
        get_absolute(cwd, lib))
}

// Find the relative path from one file to another
fn get_relative_to(abs1: path::path, abs2: path::path) -> path::path {
    assert path::path_is_absolute(abs1);
    assert path::path_is_absolute(abs2);
    #debug("finding relative path from %s to %s",
           abs1, abs2);
    let normal1 = path::normalize(abs1);
    let normal2 = path::normalize(abs2);
    let split1 = path::split(normal1);
    let split2 = path::split(normal2);
    let len1 = vec::len(split1);
    let len2 = vec::len(split2);
    assert len1 > 0u;
    assert len2 > 0u;

    let max_common_path = uint::min(len1, len2) - 1u;
    let mut start_idx = 0u;
    while start_idx < max_common_path
        && split1[start_idx] == split2[start_idx] {
        start_idx += 1u;
    }

    let mut path = []/~;
    for uint::range(start_idx, len1 - 1u) {|_i| vec::push(path, ".."); };

    path += vec::slice(split2, start_idx, len2 - 1u);

    if check vec::is_not_empty(path) {
        ret path::connect_many(path);
    } else {
        ret ".";
    }
}

fn get_absolute_rpaths(cwd: path::path, libs: [path::path]/~) -> [str]/~ {
    vec::map(libs, {|a|get_absolute_rpath(cwd, a)})
}

fn get_absolute_rpath(cwd: path::path, &&lib: path::path) -> str {
    path::dirname(get_absolute(cwd, lib))
}

fn get_absolute(cwd: path::path, lib: path::path) -> path::path {
    if path::path_is_absolute(lib) {
        lib
    } else {
        path::connect(cwd, lib)
    }
}

fn get_install_prefix_rpath(cwd: path::path, target_triple: str) -> str {
    let install_prefix = #env("CFG_PREFIX");

    if install_prefix == "" {
        fail "rustc compiled without CFG_PREFIX environment variable";
    }

    let path = [install_prefix]/~
        + filesearch::relative_target_lib_path(target_triple);
    get_absolute(cwd, path::connect_many(path))
}

fn minimize_rpaths(rpaths: [str]/~) -> [str]/~ {
    let set = map::str_hash::<()>();
    let mut minimized = []/~;
    for rpaths.each {|rpath|
        if !set.contains_key(rpath) {
            minimized += [rpath]/~;
            set.insert(rpath, ());
        }
    }
    ret minimized;
}

#[cfg(unix)]
mod test {
    #[test]
    fn test_rpaths_to_flags() {
        let flags = rpaths_to_flags(["path1", "path2"]/~);
        assert flags == ["-Wl,-rpath,path1", "-Wl,-rpath,path2"]/~;
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
        let res = get_install_prefix_rpath("/usr/lib", "triple");
        let d = path::connect(#env("CFG_PREFIX"), "/lib/rustc/triple/lib");
        assert str::ends_with(res, d);
    }

    #[test]
    fn test_prefix_rpath_abs() {
        let res = get_install_prefix_rpath("/usr/lib", "triple");
        assert path::path_is_absolute(res);
    }

    #[test]
    fn test_minimize1() {
        let res = minimize_rpaths(["rpath1", "rpath2", "rpath1"]/~);
        assert res == ["rpath1", "rpath2"]/~;
    }

    #[test]
    fn test_minimize2() {
        let res = minimize_rpaths(["1a", "2", "2", "1a", "4a",
                                   "1a", "2", "3", "4a", "3"]/~);
        assert res == ["1a", "2", "4a", "3"]/~;
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
    fn test_relative_to8() {
        let p1 = "/home/brian/Dev/rust/build/"
            + "stage2/lib/rustc/i686-unknown-linux-gnu/lib/librustc.so";
        let p2 = "/home/brian/Dev/rust/build/stage2/bin/.."
            + "/lib/rustc/i686-unknown-linux-gnu/lib/libstd.so";
        let res = get_relative_to(p1, p2);
        assert res == ".";
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_rpath_relative() {
      let o = session::os_linux;
      check not_win32(o);
      let res = get_rpath_relative_to_output(o,
            "/usr", "bin/rustc", "lib/libstd.so");
      assert res == "$ORIGIN/../lib";
    }

    #[test]
    #[cfg(target_os = "freebsd")]
    fn test_rpath_relative() {
      let o = session::os_freebsd;
      check not_win32(o);
      let res = get_rpath_relative_to_output(o,
            "/usr", "bin/rustc", "lib/libstd.so");
      assert res == "$ORIGIN/../lib";
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_rpath_relative() {
      // this is why refinements would be nice
      let o = session::os_macos;
      check not_win32(o);
      let res = get_rpath_relative_to_output(o, "/usr", "bin/rustc",
                                             "lib/libstd.so");
        assert res == "@executable_path/../lib";
    }

    #[test]
    fn test_get_absolute_rpath() {
        let res = get_absolute_rpath("/usr", "lib/libstd.so");
        assert res == "/usr/lib";
    }
}
