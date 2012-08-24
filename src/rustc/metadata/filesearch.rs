// A module for searching for libraries
// FIXME (#2658): I'm not happy how this module turned out. Should
// probably just be folded into cstore.

import result::result;
export filesearch;
export mk_filesearch;
export pick;
export pick_file;
export search;
export relative_target_lib_path;
export get_cargo_sysroot;
export get_cargo_root;
export get_cargo_root_nearest;
export libdir;

type pick<T> = fn(path: &Path) -> option<T>;

fn pick_file(file: Path, path: &Path) -> option<Path> {
    if path.file_path() == file { option::some(copy *path) }
    else { option::none }
}

trait filesearch {
    fn sysroot() -> Path;
    fn lib_search_paths() -> ~[Path];
    fn get_target_lib_path() -> Path;
    fn get_target_lib_file_path(file: &Path) -> Path;
}

fn mk_filesearch(maybe_sysroot: option<Path>,
                 target_triple: &str,
                 addl_lib_search_paths: ~[Path]) -> filesearch {
    type filesearch_impl = {sysroot: Path,
                            addl_lib_search_paths: ~[Path],
                            target_triple: ~str};
    impl filesearch_impl: filesearch {
        fn sysroot() -> Path { self.sysroot }
        fn lib_search_paths() -> ~[Path] {
            let mut paths = self.addl_lib_search_paths;

            vec::push(paths,
                      make_target_lib_path(&self.sysroot,
                                           self.target_triple));
            match get_cargo_lib_path_nearest() {
              result::ok(p) => vec::push(paths, p),
              result::err(p) => ()
            }
            match get_cargo_lib_path() {
              result::ok(p) => vec::push(paths, p),
              result::err(p) => ()
            }
            paths
        }
        fn get_target_lib_path() -> Path {
            make_target_lib_path(&self.sysroot, self.target_triple)
        }
        fn get_target_lib_file_path(file: &Path) -> Path {
            self.get_target_lib_path().push_rel(file)
        }
    }

    let sysroot = get_sysroot(maybe_sysroot);
    debug!("using sysroot = %s", sysroot.to_str());
    {sysroot: sysroot,
     addl_lib_search_paths: addl_lib_search_paths,
     target_triple: str::from_slice(target_triple)} as filesearch
}

fn search<T: copy>(filesearch: filesearch, pick: pick<T>) -> option<T> {
    let mut rslt = none;
    for filesearch.lib_search_paths().each |lib_search_path| {
        debug!("searching %s", lib_search_path.to_str());
        for os::list_dir_path(&lib_search_path).each |path| {
            debug!("testing %s", path.to_str());
            let maybe_picked = pick(path);
            if option::is_some(maybe_picked) {
                debug!("picked %s", path.to_str());
                rslt = maybe_picked;
                break;
            } else {
                debug!("rejected %s", path.to_str());
            }
        }
        if option::is_some(rslt) { break; }
    }
    return rslt;
}

fn relative_target_lib_path(target_triple: &str) -> Path {
    Path(libdir()).push_many([~"rustc",
                              str::from_slice(target_triple),
                              libdir()])
}

fn make_target_lib_path(sysroot: &Path,
                        target_triple: &str) -> Path {
    sysroot.push_rel(&relative_target_lib_path(target_triple))
}

fn get_default_sysroot() -> Path {
    match os::self_exe_path() {
      option::some(p) => p.pop(),
      option::none => fail ~"can't determine value for sysroot"
    }
}

fn get_sysroot(maybe_sysroot: option<Path>) -> Path {
    match maybe_sysroot {
      option::some(sr) => sr,
      option::none => get_default_sysroot()
    }
}

fn get_cargo_sysroot() -> result<Path, ~str> {
    result::ok(get_default_sysroot().push_many([libdir(), ~"cargo"]))
}

fn get_cargo_root() -> result<Path, ~str> {
    match os::getenv(~"CARGO_ROOT") {
        some(_p) => result::ok(Path(_p)),
        none => match os::homedir() {
          some(_q) => result::ok(_q.push(".cargo")),
          none => result::err(~"no CARGO_ROOT or home directory")
        }
    }
}

fn get_cargo_root_nearest() -> result<Path, ~str> {
    do result::chain(get_cargo_root()) |p| {
        let cwd = os::getcwd();
        let cwd_cargo = cwd.push(".cargo");
        let mut par_cargo = cwd.pop().push(".cargo");
        let mut rslt = result::ok(cwd_cargo);

        if !os::path_is_dir(&cwd_cargo) && cwd_cargo != p {
            while par_cargo != p {
                if os::path_is_dir(&par_cargo) {
                    rslt = result::ok(par_cargo);
                    break;
                }
                if par_cargo.components.len() == 1 {
                    // We just checked /.cargo, stop now.
                    break;
                }
                par_cargo = par_cargo.pop().pop().push(".cargo");
            }
        }
        rslt
    }
}

fn get_cargo_lib_path() -> result<Path, ~str> {
    do result::chain(get_cargo_root()) |p| {
        result::ok(p.push(libdir()))
    }
}

fn get_cargo_lib_path_nearest() -> result<Path, ~str> {
    do result::chain(get_cargo_root_nearest()) |p| {
        result::ok(p.push(libdir()))
    }
}

// The name of the directory rustc expects libraries to be located.
// On Unix should be "lib", on windows "bin"
fn libdir() -> ~str {
   let libdir = env!("CFG_LIBDIR");
   if str::is_empty(libdir) {
      fail ~"rustc compiled without CFG_LIBDIR environment variable";
   }
   libdir
}
