import std::option;
import std::fs;
import std::vec;
import std::str;
import back::link;

export filesearch;
export mk_filesearch;

type filesearch = obj {
    fn sysroot() -> fs::path;
    fn lib_search_paths() -> [fs::path];
    fn get_target_lib_path() -> fs::path;
    fn get_target_lib_file_path(file: fs::path) -> fs::path;
};

fn mk_filesearch(binary_name: fs::path,
                 maybe_sysroot: option::t<fs::path>,
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

    let sysroot = get_sysroot(maybe_sysroot, binary_name);
    ret filesearch_impl(sysroot, addl_lib_search_paths, target_triple);
}

fn make_target_lib_path(sysroot: fs::path,
                        target_triple: str) -> fs::path {
    let path = [sysroot, "lib/rustc", target_triple, "lib"];
    check vec::is_not_empty(path);
    let path = fs::connect_many(path);
    ret path;
}

fn get_default_sysroot(binary: fs::path) -> fs::path {
    let dirname = fs::dirname(binary);
    if str::eq(dirname, binary) { ret "../"; }
    ret fs::connect(dirname, "../");
}

fn get_sysroot(maybe_sysroot: option::t<fs::path>,
               binary: fs::path) -> fs::path {
    alt maybe_sysroot {
      option::some(sr) { sr }
      option::none. { get_default_sysroot(binary) }
    }
}