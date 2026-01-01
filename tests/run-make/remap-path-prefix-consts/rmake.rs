//@ ignore-cross-compile (relocations in generic ELF against `arm-unknown-linux-gnueabihf`)
//@ needs-target-std

use run_make_support::{bin_name, cwd, run, rustc};

fn main() {
    // No remapping - relative paths
    {
        let runner_bin = bin_name("runner-no-remap-rel-paths");

        let mut location_caller = rustc();
        location_caller.crate_type("lib").input("location-caller.rs");
        location_caller.run();

        let mut runner = rustc();
        runner.crate_type("bin").input("runner.rs").output(&runner_bin);
        runner.run();

        run(&runner_bin);
    }

    // No remapping - absolute paths
    {
        let runner_bin = bin_name("runner-no-remap-abs-paths");

        let mut location_caller = rustc();
        location_caller.crate_type("lib").input(cwd().join("location-caller.rs"));
        location_caller.run();

        let mut runner = rustc();
        runner.crate_type("bin").input(cwd().join("runner.rs")).output(&runner_bin);
        runner.run();

        run(&runner_bin);
    }

    // No remapping - mixed paths
    {
        let runner_bin = bin_name("runner-no-remap-mixed-paths");

        let mut location_caller = rustc();
        location_caller.crate_type("lib").input(cwd().join("location-caller.rs"));
        location_caller.run();

        let mut runner = rustc();
        runner.crate_type("bin").input("runner.rs").output(&runner_bin);
        runner.run();

        run(&runner_bin);
    }

    // Remapping current working directory
    {
        let runner_bin = bin_name("runner-remap-cwd");

        let mut location_caller = rustc();
        location_caller
            .crate_type("lib")
            .remap_path_prefix(cwd(), "/remapped")
            .input(cwd().join("location-caller.rs"));
        location_caller.run();

        let mut runner = rustc();
        runner
            .crate_type("bin")
            .remap_path_prefix(cwd(), "/remapped")
            .input(cwd().join("runner.rs"))
            .output(&runner_bin);
        runner.run();

        run(&runner_bin);
    }

    // Remapping current working directory - only in the dependency
    {
        let runner_bin = bin_name("runner-remap-cwd-only-dep");

        let mut location_caller = rustc();
        location_caller
            .crate_type("lib")
            .remap_path_prefix(cwd(), "/remapped")
            .input(cwd().join("location-caller.rs"));
        location_caller.run();

        let mut runner = rustc();
        runner.crate_type("bin").input(cwd().join("runner.rs")).output(&runner_bin);
        runner.run();

        run(&runner_bin);
    }

    // Remapping current working directory - different scopes
    {
        let runner_bin = bin_name("runner-remap-cwd-diff-scope");

        let mut location_caller = rustc();
        location_caller
            .crate_type("lib")
            .remap_path_prefix(cwd(), "/remapped")
            .arg("-Zremap-path-scope=object")
            .input(cwd().join("location-caller.rs"));
        location_caller.run();

        let mut runner = rustc();
        runner
            .crate_type("bin")
            .remap_path_prefix(cwd(), "/remapped")
            .arg("-Zremap-path-scope=diagnostics")
            .input(cwd().join("runner.rs"))
            .output(&runner_bin);
        runner.run();

        run(&runner_bin);
    }
}
