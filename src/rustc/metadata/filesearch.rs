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

import path::path;

type pick<T> = fn(path: path) -> option<T>;

fn pick_file(file: path, path: path) -> option<path> {
    if path::basename(path) == file { option::some(path) }
    else { option::none }
}

iface filesearch {
    fn sysroot() -> path;
    fn lib_search_paths() -> ~[path];
    fn get_target_lib_path() -> path;
    fn get_target_lib_file_path(file: path) -> path;
}

fn mk_filesearch(maybe_sysroot: option<path>,
                 target_triple: str,
                 addl_lib_search_paths: ~[path]) -> filesearch {
    type filesearch_impl = {sysroot: path,
                            addl_lib_search_paths: ~[path],
                            target_triple: str};
    impl of filesearch for filesearch_impl {
        fn sysroot() -> path { self.sysroot }
        fn lib_search_paths() -> ~[path] {
            let mut paths = self.addl_lib_search_paths;

            vec::push(paths,
                      make_target_lib_path(self.sysroot, self.target_triple));
            alt get_cargo_lib_path_nearest() {
              result::ok(p) { vec::push(paths, p) }
              result::err(p) { }
            }
            alt get_cargo_lib_path() {
              result::ok(p) { vec::push(paths, p) }
              result::err(p) { }
            }
            paths
        }
        fn get_target_lib_path() -> path {
            make_target_lib_path(self.sysroot, self.target_triple)
        }
        fn get_target_lib_file_path(file: path) -> path {
            path::connect(self.get_target_lib_path(), file)
        }
    }

    let sysroot = get_sysroot(maybe_sysroot);
    #debug("using sysroot = %s", sysroot);
    {sysroot: sysroot,
     addl_lib_search_paths: addl_lib_search_paths,
     target_triple: target_triple} as filesearch
}

fn search<T: copy>(filesearch: filesearch, pick: pick<T>) -> option<T> {
    let mut rslt = none;
    for filesearch.lib_search_paths().each |lib_search_path| {
        #debug("searching %s", lib_search_path);
        for os::list_dir_path(lib_search_path).each |path| {
            #debug("testing %s", path);
            let maybe_picked = pick(path);
            if option::is_some(maybe_picked) {
                #debug("picked %s", path);
                rslt = maybe_picked;
                break;
            } else {
                #debug("rejected %s", path);
            }
        }
        if option::is_some(rslt) { break; }
    }
    ret rslt;
}

fn relative_target_lib_path(target_triple: str) -> ~[path] {
    ~[libdir(), "rustc", target_triple, libdir()]
}

fn make_target_lib_path(sysroot: path,
                        target_triple: str) -> path {
    let path = vec::append(~[sysroot],
                           relative_target_lib_path(target_triple));
    let path = path::connect_many(path);
    ret path;
}

fn get_default_sysroot() -> path {
    alt os::self_exe_path() {
      option::some(p) { path::normalize(path::connect(p, "..")) }
      option::none {
        fail "can't determine value for sysroot";
      }
    }
}

fn get_sysroot(maybe_sysroot: option<path>) -> path {
    alt maybe_sysroot {
      option::some(sr) { sr }
      option::none { get_default_sysroot() }
    }
}

fn get_cargo_sysroot() -> result<path, str> {
    let path = ~[get_default_sysroot(), libdir(), "cargo"];
    result::ok(path::connect_many(path))
}

fn get_cargo_root() -> result<path, str> {
    alt os::getenv("CARGO_ROOT") {
        some(_p) { result::ok(_p) }
        none {
          alt os::homedir() {
            some(_q) { result::ok(path::connect(_q, ".cargo")) }
            none { result::err("no CARGO_ROOT or home directory") }
          }
        }
    }
}

fn get_cargo_root_nearest() -> result<path, str> {
    do result::chain(get_cargo_root()) |p| {
        let cwd = os::getcwd();
        let mut dirname = path::dirname(cwd);
        let mut dirpath = path::split(dirname);
        let cwd_cargo = path::connect(cwd, ".cargo");
        let mut par_cargo = path::connect(dirname, ".cargo");
        let mut rslt = result::ok(cwd_cargo);

        if !os::path_is_dir(cwd_cargo) && cwd_cargo != p {
            while vec::is_not_empty(dirpath) && par_cargo != p {
                if os::path_is_dir(par_cargo) {
                    rslt = result::ok(par_cargo);
                    break;
                }
                vec::pop(dirpath);
                dirname = path::dirname(dirname);
                par_cargo = path::connect(dirname, ".cargo");
            }
        }
        rslt
    }
}

fn get_cargo_lib_path() -> result<path, str> {
    do result::chain(get_cargo_root()) |p| {
        result::ok(path::connect(p, libdir()))
    }
}

fn get_cargo_lib_path_nearest() -> result<path, str> {
    do result::chain(get_cargo_root_nearest()) |p| {
        result::ok(path::connect(p, libdir()))
    }
}

// The name of the directory rustc expects libraries to be located.
// On Unix should be "lib", on windows "bin"
fn libdir() -> str {
   let libdir = #env("CFG_LIBDIR");
   if str::is_empty(libdir) {
      fail "rustc compiled without CFG_LIBDIR environment variable";
   }
   libdir
}
