// A module for searching for libraries
// FIXME: I'm not happy how this module turned out. Should probably
// just be folded into cstore.

import core::option;
import std::fs;
import vec;
import std::os;
import result;
import std::generic_os;

export filesearch;
export mk_filesearch;
export pick;
export pick_file;
export search;
export relative_target_lib_path;
export get_cargo_root;

type pick<T> = block(path: fs::path) -> option::t<T>;

fn pick_file(file: fs::path, path: fs::path) -> option::t<fs::path> {
    if fs::basename(path) == file { option::some(path) }
    else { option::none }
}

iface filesearch {
    fn sysroot() -> fs::path;
    fn lib_search_paths() -> [fs::path];
    fn get_target_lib_path() -> fs::path;
    fn get_target_lib_file_path(file: fs::path) -> fs::path;
}

fn mk_filesearch(maybe_sysroot: option::t<fs::path>,
                 target_triple: str,
                 addl_lib_search_paths: [fs::path]) -> filesearch {
    type filesearch_impl = {sysroot: fs::path,
                            addl_lib_search_paths: [fs::path],
                            target_triple: str};
    impl of filesearch for filesearch_impl {
        fn sysroot() -> fs::path { self.sysroot }
        fn lib_search_paths() -> [fs::path] {
            self.addl_lib_search_paths
                + [make_target_lib_path(self.sysroot, self.target_triple)]
                + alt get_cargo_lib_path() {
                  result::ok(p) { [p] }
                  result::err(p) { [] }
                }
        }
        fn get_target_lib_path() -> fs::path {
            make_target_lib_path(self.sysroot, self.target_triple)
        }
        fn get_target_lib_file_path(file: fs::path) -> fs::path {
            fs::connect(self.get_target_lib_path(), file)
        }
    }

    let sysroot = get_sysroot(maybe_sysroot);
    #debug("using sysroot = %s", sysroot);
    {sysroot: sysroot,
     addl_lib_search_paths: addl_lib_search_paths,
     target_triple: target_triple} as filesearch
}

// FIXME #1001: This can't be an obj method
fn search<T: copy>(filesearch: filesearch, pick: pick<T>) -> option::t<T> {
    for lib_search_path in filesearch.lib_search_paths() {
        #debug("searching %s", lib_search_path);
        for path in fs::list_dir(lib_search_path) {
            #debug("testing %s", path);
            let maybe_picked = pick(path);
            if option::is_some(maybe_picked) {
                #debug("picked %s", path);
                ret maybe_picked;
            } else {
                #debug("rejected %s", path);
            }
        }
    }
    ret option::none;
}

fn relative_target_lib_path(target_triple: str) -> [fs::path] {
    ["lib", "rustc", target_triple, "lib"]
}

fn make_target_lib_path(sysroot: fs::path,
                        target_triple: str) -> fs::path {
    let path = [sysroot] + relative_target_lib_path(target_triple);
    check vec::is_not_empty(path);
    let path = fs::connect_many(path);
    ret path;
}

fn get_default_sysroot() -> fs::path {
    alt os::get_exe_path() {
      option::some(p) { fs::normalize(fs::connect(p, "..")) }
      option::none. {
        fail "can't determine value for sysroot";
      }
    }
}

fn get_sysroot(maybe_sysroot: option::t<fs::path>) -> fs::path {
    alt maybe_sysroot {
      option::some(sr) { sr }
      option::none. { get_default_sysroot() }
    }
}

fn get_cargo_root() -> result::t<fs::path, str> {
    alt generic_os::getenv("CARGO_ROOT") {
        some(_p) { result::ok(_p) }
        none. {
          alt fs::homedir() {
            some(_q) { result::ok(fs::connect(_q, ".cargo")) }
            none. { result::err("no CARGO_ROOT or home directory") }
          }
        }
    }
}

fn get_cargo_lib_path() -> result::t<fs::path, str> {
    result::chain(get_cargo_root()) { |p|
        result::ok(fs::connect(p, "lib"))
    }
}