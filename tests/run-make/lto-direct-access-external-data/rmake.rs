//@ needs-llvm-components: x86
//@ ignore-loongarch64 (handles dso_local differently)
//@ ignore-powerpc64 (handles dso_local differently)
//@ ignore-apple (handles dso_local differently)

// Test the various interleavings of LTO and relocation model.
// We know that in some cases the -Zdirect-access-external-data option will not
// result in the correct code generation due to the module having
// inconsistently set PIE Level. Test those cases so that the result is clear
// when future patches fix the problem.

use run_make_support::{
    cwd, has_extension, llvm_dis, llvm_filecheck, llvm_readobj, rfs, rustc, shallow_find_files,
};

struct TestCase {
    lto: &'static str,
    reloc: &'static str,
    direct_access: Option<&'static str>,
    expected_prefix: &'static str,
    dep_prefixes: &'static [&'static str],
}

fn main() {
    let test_cases = [
        TestCase {
            lto: "thin",
            reloc: "static",
            direct_access: None,
            expected_prefix: "DEFAULT",
            dep_prefixes: &[],
        },
        TestCase {
            lto: "fat",
            reloc: "static",
            direct_access: None,
            expected_prefix: "DEFAULT",
            dep_prefixes: &[],
        },
        TestCase {
            lto: "thin",
            reloc: "pie",
            direct_access: None,
            expected_prefix: "PIE",
            dep_prefixes: &["DEP-NO-PIE"],
        },
        TestCase {
            lto: "fat",
            reloc: "pie",
            direct_access: None,
            expected_prefix: "PIE",
            dep_prefixes: &[],
        },
        TestCase {
            lto: "thin",
            reloc: "pie",
            direct_access: Some("yes"),
            expected_prefix: "DIRECT",
            dep_prefixes: &["DEP-NO-PIE"],
        },
        TestCase {
            lto: "fat",
            reloc: "pie",
            direct_access: Some("yes"),
            expected_prefix: "DIRECT",
            dep_prefixes: &[],
        },
        TestCase {
            lto: "thin",
            reloc: "static",
            direct_access: Some("no"),
            expected_prefix: "INDIRECT",
            dep_prefixes: &[],
        },
        TestCase {
            lto: "fat",
            reloc: "static",
            direct_access: Some("no"),
            expected_prefix: "INDIRECT",
            dep_prefixes: &[],
        },
    ];

    for case in test_cases {
        // Remove all output files that are not source Rust code for cleanup.
        for file in shallow_find_files(cwd(), |path| !has_extension(path, "rs")) {
            rfs::remove_file(file);
        }

        let mut cmd = rustc();
        cmd.input("dep.rs")
            .target("x86_64-unknown-linux-gnu")
            .crate_type("rlib")
            .crate_name("dep")
            .arg("-Cpanic=abort");

        if let Some(da) = case.direct_access {
            cmd.arg(format!("-Zdirect-access-external-data={da}"));
        }
        cmd.run();

        cmd = rustc();
        cmd.input("lib.rs")
            .target("x86_64-unknown-linux-gnu")
            .crate_type("staticlib")
            .crate_name("lto_direct_access_test")
            .extern_("dep", "libdep.rlib")
            .arg("-Cpanic=abort")
            .arg("-Csave-temps")
            .arg(format!("-Clto={}", case.lto))
            .arg(format!("-Crelocation-model={}", case.reloc))
            .arg("-Ccodegen-units=2");

        if let Some(da) = case.direct_access {
            cmd.arg(format!("-Zdirect-access-external-data={da}"));
        }
        cmd.run();

        let suffix = if case.lto == "thin" {
            ".rcgu.thin-lto-after-pm.bc"
        } else {
            ".rcgu.lto.after-restriction.bc"
        };

        let bc_files = shallow_find_files(".", |path| {
            let name = path.file_name().unwrap().to_str().unwrap();
            name.starts_with("lto_direct_access_test.lto_direct_access_test.")
                && name.ends_with(suffix)
        });
        assert!(!bc_files.is_empty(), "No expected bitcode files were generated");

        for bc_file in &bc_files {
            llvm_dis().input(bc_file).run();
            let ll_file = bc_file.with_extension("ll");
            llvm_filecheck()
                .input_file(&ll_file)
                .patterns("lib.rs")
                .check_prefix(case.expected_prefix)
                .run();
        }

        let relocs = llvm_readobj().arg("--relocations").input("liblto_direct_access_test.a").run();
        let relocs_out = "relocs.txt";
        rfs::write(relocs_out, relocs.stdout_utf8());

        let reloc_prefix = match case.expected_prefix {
            "DIRECT" => {
                if case.lto == "thin" {
                    Some("DIRECT-RELOC-THIN")
                } else {
                    Some("DIRECT-RELOC-FAT")
                }
            }
            "PIE" => Some("PIE-RELOC"),
            "INDIRECT" => Some("INDIRECT-RELOC"),
            _ => None,
        };

        if let Some(prefix) = reloc_prefix {
            llvm_filecheck().input_file(relocs_out).patterns("lib.rs").check_prefix(prefix).run();
        }

        if !case.dep_prefixes.is_empty() {
            let dep_bc_files = shallow_find_files(".", |path| {
                let name = path.file_name().unwrap().to_str().unwrap();
                name.contains("dep") && name.ends_with(suffix)
            });
            assert!(!dep_bc_files.is_empty(), "dep bitcode files not found");
            for dep_bc in &dep_bc_files {
                llvm_dis().input(dep_bc).run();
                let dep_ll = dep_bc.with_extension("ll");
                for dep_cp in case.dep_prefixes {
                    llvm_filecheck()
                        .input_file(&dep_ll)
                        .patterns("lib.rs")
                        .check_prefix(dep_cp)
                        .run();
                }
            }
        }
    }
}
