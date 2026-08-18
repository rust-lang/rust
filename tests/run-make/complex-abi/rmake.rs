//@ needs-target-std
//@ ignore-android: FIXME(#142855)
//@ ignore-sgx: (x86 machine code cannot be directly executed)
//@ ignore-pauthtest: (it requires non-trivial compilation of c sources, and only supports dynamic
//  linking, ignore the test).

use run_make_support::{bin_name, build_native_static_lib, env_var, run, rustc};

fn main() {
    build_native_static_lib("test");
    rustc().linker(&env_var("CC")).input("main.rs").run();
    run(&bin_name("main"));
}
