// A module for searching for libraries
// FIXME: I'm not happy how this module turned out. Should probably
// just be folded into cstore.

import std::option;
import std::fs;
import std::vec;
import std::str;
import std::os;
import back::link;

export filesearch;
export mk_filesearch;
export pick;
export pick_file;
export search;
export relative_target_lib_path;

type pick<@T> = block(path: fs::path) -> option::t<T>;

fn pick_file(file: fs::path, path: fs::path) -> option::t<fs::path> {
    if fs::basename(path) == file { option::some(path) }
    else { option::none }
}

type filesearch = obj {
    fn sysroot() -> fs::path;
    fn lib_search_paths() -> [fs::path];
    fn get_target_lib_path() -> fs::path;
    fn get_target_lib_file_path(file: fs::path) -> fs::path;
};

fn mk_filesearch(maybe_sysroot: option::t<fs::path>,
                 target_triple: str,
                 addl_lib_search_paths: [fs::path]) -> filesearch {
    obj filesearch_impl(sysroot: fs::path,
                        addl_lib_search_paths: [fs::path],
                        target_triple: str) {
        fn sysroot() -> fs::path { sysroot }
        fn lib_search_paths() -> [fs::path] {
            addl_lib_search_paths
                + [make_target_lib_path(sysroot, target_triple)]
        }

        fn get_target_lib_path() -> fs::path {
            make_target_lib_path(sysroot, target_triple)
        }

        fn get_target_lib_file_path(file: fs::path) -> fs::path {
            fs::connect(self.get_target_lib_path(), file)
        }
    }

    let sysroot = get_sysroot(maybe_sysroot);
    log #fmt("using sysroot = %s", sysroot);
    ret filesearch_impl(sysroot, addl_lib_search_paths, target_triple);
}

// FIXME #1001: This can't be an obj method
fn search<@T>(filesearch: filesearch, pick: pick<T>) -> option::t<T> {
    for lib_search_path in filesearch.lib_search_paths() {
        log #fmt["searching %s", lib_search_path];
        for path in fs::list_dir(lib_search_path) {
            log #fmt["testing %s", path];
            let maybe_picked = pick(path);
            if option::is_some(maybe_picked) {
                log #fmt("picked %s", path);
                ret maybe_picked;
            } else {
                log #fmt("rejected %s", path);
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