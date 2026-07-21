// Check incremental reuse of the synthetic fat core. Reuse requires both a
// matching persisted merge and LLVM's ThinLTO key for the core and its imports.

//@ ignore-cross-compile
//@ needs-target-std

use run_make_support::{rfs, run, rust_lib_name, rustc};

const CORE_REUSED: &str = " - __rustc_fat_lto_core: re-used";
const CORE_RECOMPILED: &str = " - __rustc_fat_lto_core: re-compiled";
const CORE_PIPELINE: &str = "running thin lto passes over __rustc_fat_lto_core";
const CORE_MERGE: &str = "going for a fat lto";
const PRE_LTO_HIT: &str = "fat LTO core pre-LTO cache hit";
const PRE_LTO_MISS: &str = "fat LTO core pre-LTO cache miss";

fn compile_deps(hot_b: bool, tail_b: bool) {
    let mut hot = rustc();
    hot.input("hot.rs")
        .crate_name("hot")
        .crate_type("rlib")
        .edition("2021")
        .opt_level("3")
        // Keep member names stable so the body edit isolates LLVM's content hash.
        .metadata("fixed-hot")
        // Use a file name unrelated to the crate name to ensure core membership
        // comes from crate metadata rather than archive-member text.
        .output(rust_lib_name("renamed_hot"));
    if hot_b {
        hot.cfg("hot_b");
    }
    hot.run();

    let mut tail = rustc();
    tail.input("tail.rs")
        .crate_name("tail")
        .crate_type("rlib")
        .edition("2021")
        // Keep the crate disambiguator stable so this fixture changes only
        // the imported function body, not the symbol referenced by the core.
        .metadata("fixed-tail")
        .opt_level("3")
        .output(rust_lib_name("renamed_tail"));
    if tail_b {
        tail.cfg("tail_b");
    }
    tail.run();
}

fn build(out: &str, incr: &str, partitions: usize, copy_debug: bool) -> String {
    let mut cmd = rustc();
    cmd.input("main.rs")
        .crate_name("main")
        .edition("2021")
        .extern_("hot", rust_lib_name("renamed_hot"))
        .extern_("tail", rust_lib_name("renamed_tail"))
        .opt_level("3")
        .codegen_units(2)
        .lto("thin")
        .arg("-Zfat-lto-crates=hot,main")
        .arg("-Zverify-llvm-ir")
        .incremental(incr)
        // The output path affects the entry-point CGU's LLVM hash, so use one
        // compiler path and copy each result to a snapshot.
        .output("main-output");
    if partitions > 1 {
        cmd.arg(format!("-Zfat-lto-partitions={partitions}"));
    }
    let log = if copy_debug {
        "rustc_codegen_llvm::back::lto=info,rustc_codegen_ssa::back::write=debug"
    } else {
        "rustc_codegen_llvm::back::lto=info"
    };
    let stderr = cmd.env("RUSTC_LOG", log).run().stderr_utf8();
    rfs::copy("main-output", out);
    stderr
}

fn assert_reused(log: &str) {
    assert!(log.contains(CORE_REUSED), "core was not reused:\n{log}");
    assert!(log.contains(PRE_LTO_HIT), "merged core bitcode was not reused:\n{log}");
    assert!(!log.contains(CORE_MERGE), "a reused core was merged again:\n{log}");
    assert!(
        !log.contains(CORE_PIPELINE),
        "a reused core unexpectedly ran the fat post-link pipeline:\n{log}"
    );
}

fn assert_recompiled(log: &str) {
    assert!(log.contains(CORE_RECOMPILED), "core was not recompiled:\n{log}");
    assert!(!log.contains(CORE_REUSED), "core was also reported as reused:\n{log}");
    assert!(log.contains(CORE_PIPELINE), "core pipeline did not run:\n{log}");
}

fn files_named(root: &std::path::Path, name: &str, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files_named(&path, name, out);
        } else if path.file_name().unwrap() == name {
            out.push(path);
        }
    }
}

fn newest_file_named(root: &str, name: &str) -> std::path::PathBuf {
    let mut files = Vec::new();
    files_named(std::path::Path::new(root), name, &mut files);
    files.into_iter().max().unwrap_or_else(|| panic!("did not find {name} under {root}"))
}

fn main() {
    compile_deps(false, false);

    // The first cached representation can differ from the just-generated input.
    // Allow one convergence build; the third must copy the post-LTO object.
    let first_log = build("main-a1", "incr", 1, false);
    assert!(
        first_log.contains("renamed_hot.hot") && first_log.contains("to fat LTO core"),
        "renamed upstream rlib was not selected by crate identity:\n{first_log}"
    );
    build("main-a2", "incr", 1, false);
    let a3_log = build("main-a3", "incr", 1, false);
    assert_reused(&a3_log);
    let a_stdout = run("main-a3").stdout_utf8();
    let a_binary = rfs::read("main-a3");
    assert_eq!(rfs::read("main-a1"), a_binary);
    assert_eq!(rfs::read("main-a2"), a_binary);

    // A core body edit must miss both cache layers and match a fresh build byte for byte.
    compile_deps(true, false);
    let hot_b_log = build("main-hot-b", "incr", 1, false);
    assert_recompiled(&hot_b_log);
    assert!(hot_b_log.contains(PRE_LTO_MISS));
    let hot_b_stdout = run("main-hot-b").stdout_utf8();
    assert_ne!(hot_b_stdout, a_stdout);
    build("main-hot-b-fresh", "fresh-hot-b", 1, false);
    assert_eq!(rfs::read("main-hot-b"), rfs::read("main-hot-b-fresh"));

    // Returning to A after B must not reuse B's object: rebuild once, then reuse A.
    compile_deps(false, false);
    let back_to_a_log = build("main-a4", "incr", 1, false);
    assert_recompiled(&back_to_a_log);
    assert!(back_to_a_log.contains(PRE_LTO_MISS));
    assert_eq!(rfs::read("main-a4"), a_binary);
    let a5_log = build("main-a5", "incr", 1, false);
    assert_reused(&a5_log);

    // Corrupt the persisted merge. Its BLAKE3 checksum must force a rebuild before parsing.
    let cache_path = newest_file_named("incr", "fat-lto-core.pre-lto.bc");
    let mut damaged = rfs::read(&cache_path);
    let last = damaged.last_mut().unwrap();
    *last ^= 0xff;
    rfs::write(&cache_path, damaged);
    let damaged_log = build("main-damaged", "incr", 1, false);
    assert_recompiled(&damaged_log);
    assert!(damaged_log.contains(PRE_LTO_MISS));
    assert_eq!(rfs::read("main-damaged"), a_binary);
    assert_reused(&build("main-after-damage", "incr", 1, false));

    // A malformed ThinLTO key file must cause a cache miss, not a compiler crash.
    let key_path = newest_file_named("incr", "thin-lto-past-keys.bin");
    rfs::write(&key_path, b"malformed\n");
    let malformed_key_log = build("main-malformed-key", "incr", 1, false);
    assert_recompiled(&malformed_key_log);
    assert!(malformed_key_log.contains(PRE_LTO_HIT));
    assert_eq!(rfs::read("main-malformed-key"), a_binary);
    assert_reused(&build("main-after-malformed-key", "incr", 1, false));

    // Duplicate records are corrupt even when they agree. Reject the file
    // instead of choosing an arbitrary occurrence.
    let key_path = newest_file_named("incr", "thin-lto-past-keys.bin");
    let valid_keys = rfs::read(&key_path);
    rfs::write(&key_path, [valid_keys.as_slice(), valid_keys.as_slice()].concat());
    let duplicate_key_log = build("main-duplicate-key", "incr", 1, false);
    assert_recompiled(&duplicate_key_log);
    assert!(duplicate_key_log.contains(PRE_LTO_HIT));
    assert_eq!(rfs::read("main-duplicate-key"), a_binary);
    assert_reused(&build("main-after-duplicate-key", "incr", 1, false));

    // Change only an always-inline function imported from the non-core tail. The
    // pre-LTO merge must hit, while LLVM's import-sensitive key rejects the object.
    compile_deps(false, true);
    let tail_b_log = build("main-tail-b", "incr", 1, false);
    assert_recompiled(&tail_b_log);
    assert!(tail_b_log.contains(PRE_LTO_HIT));
    assert_ne!(run("main-tail-b").stdout_utf8(), a_stdout);
    build("main-tail-b-fresh", "fresh-tail-b", 1, false);
    assert_eq!(rfs::read("main-tail-b"), rfs::read("main-tail-b-fresh"));

    // Restore the imported body; the core must rebuild A rather than reuse B.
    compile_deps(false, false);
    let tail_a_log = build("main-tail-a", "incr", 1, false);
    assert_recompiled(&tail_a_log);
    assert!(tail_a_log.contains(PRE_LTO_HIT));
    assert_eq!(rfs::read("main-tail-a"), a_binary);

    // A partitioned core is reusable only as a complete set. Verify that reuse
    // copies every function/data partition.
    build("main-p1", "incr-partitioned", 2, false);
    build("main-p2", "incr-partitioned", 2, false);
    let partitioned_binary = rfs::read("main-p1");
    assert_eq!(rfs::read("main-p2"), partitioned_binary);
    let partitioned_log = build("main-p3", "incr-partitioned", 2, true);
    assert_reused(&partitioned_log);
    let copied_partitions = partitioned_log
        .matches("copying preexisting module `__rustc_fat_lto_core.fat-lto-part")
        .count();
    assert_eq!(copied_partitions, 3, "did not copy the complete partition set");
    assert_eq!(run("main-p3").stdout_utf8(), a_stdout);
    assert_eq!(rfs::read("main-p3"), partitioned_binary);

    // Remove one partition. The pre-LTO merge still hits, but all post-LTO
    // partitions must be regenerated.
    let missing_partition =
        newest_file_named("incr-partitioned", "__rustc_fat_lto_core.fat-lto-part0001.o");
    rfs::remove_file(missing_partition);
    let incomplete_log = build("main-p4", "incr-partitioned", 2, false);
    assert_recompiled(&incomplete_log);
    assert!(incomplete_log.contains(PRE_LTO_HIT));
    assert_eq!(rfs::read("main-p4"), partitioned_binary);
    let repaired_log = build("main-p5", "incr-partitioned", 2, true);
    assert_reused(&repaired_log);
    assert_eq!(
        repaired_log
            .matches("copying preexisting module `__rustc_fat_lto_core.fat-lto-part")
            .count(),
        3
    );
    assert_eq!(rfs::read("main-p5"), partitioned_binary);

    // A tracked partition-count change invalidates the complete output set;
    // a matching name prefix must not admit stale partitions.
    let repartitioned_log = build("main-p6", "incr-partitioned", 3, true);
    assert_recompiled(&repartitioned_log);
    assert!(!repartitioned_log.contains("copying preexisting module `__rustc_fat_lto_core"));
    assert_eq!(run("main-p6").stdout_utf8(), a_stdout);
    let repartitioned_reuse_log = build("main-p7", "incr-partitioned", 3, true);
    assert_reused(&repartitioned_reuse_log);
    assert_eq!(
        repartitioned_reuse_log
            .matches("copying preexisting module `__rustc_fat_lto_core.fat-lto-part")
            .count(),
        4
    );

    // Shrinking the partition count must reject the matching prefix of the old set.
    let smaller_log = build("main-p8", "incr-partitioned", 2, true);
    assert_recompiled(&smaller_log);
    assert!(!smaller_log.contains("copying preexisting module `__rustc_fat_lto_core"));
    assert_eq!(rfs::read("main-p8"), partitioned_binary);
    let smaller_reuse_log = build("main-p9", "incr-partitioned", 2, true);
    assert_reused(&smaller_reuse_log);
    assert_eq!(
        smaller_reuse_log
            .matches("copying preexisting module `__rustc_fat_lto_core.fat-lto-part")
            .count(),
        3
    );
}
