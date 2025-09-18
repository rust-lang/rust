//! These tests use insta for snapshot testing.
//! See bootstrap's README on how to bless the snapshots.

use std::path::PathBuf;

use crate::core::build_steps::{compile, dist, doc, test, tool};
use crate::core::builder::tests::{
    RenderConfig, TEST_TRIPLE_1, TEST_TRIPLE_2, TEST_TRIPLE_3, configure, first, host_target,
    render_steps, run_build,
};
use crate::core::builder::{Builder, Kind, StepDescription, StepMetadata};
use crate::core::config::TargetSelection;
use crate::core::config::toml::rust::with_lld_opt_in_targets;
use crate::utils::cache::Cache;
use crate::utils::helpers::get_host_target;
use crate::utils::tests::{ConfigBuilder, TestCtx};
use crate::{Build, Compiler, Config, Flags, Subcommand};

#[test]
fn build_default() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustdoc 1 <host>
    ");
}

#[test]
fn build_cross_compile() {
    let ctx = TestCtx::new();

    insta::assert_snapshot!(
        ctx.config("build")
            // Cross-compilation fails on stage 1, as we don't have a stage0 std available
            // for non-host targets.
            .stage(2)
            .hosts(&[&host_target(), TEST_TRIPLE_1])
            .targets(&[&host_target(), TEST_TRIPLE_1])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 2 <host> -> std 2 <host>
    [build] rustc 1 <host> -> std 1 <target1>
    [build] rustc 2 <host> -> std 2 <target1>
    [build] rustdoc 2 <host>
    [build] llvm <target1>
    [build] rustc 1 <host> -> rustc 2 <target1>
    [build] rustdoc 2 <target1>
    ");
}

#[test]
fn build_with_empty_host() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx
            .config("build")
            .hosts(&[])
            .targets(&[TEST_TRIPLE_1])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <target1>
    "
    );
}

#[test]
fn build_compiler_no_explicit_stage() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("compiler")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    ");
}

#[test]
fn build_rustc_no_explicit_stage() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("rustc")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    ");
}

#[test]
#[should_panic]
fn build_compiler_stage_0() {
    let ctx = TestCtx::new();
    ctx.config("build").path("compiler").stage(0).run();
}

#[test]
fn build_compiler_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("compiler")
            .stage(1)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    ");
}

#[test]
fn build_compiler_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("compiler")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    ");
}

#[test]
fn build_compiler_stage_3() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("compiler")
            .stage(3)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 2 <host> -> std 2 <host>
    [build] rustc 2 <host> -> rustc 3 <host>
    ");
}

#[test]
fn build_compiler_stage_3_cross() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("compiler")
            .hosts(&[TEST_TRIPLE_1])
            .stage(3)
            .render_steps(), @r"
    [build] llvm <host>
    [build] llvm <target1>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> std 1 <target1>
    [build] rustc 2 <host> -> std 2 <target1>
    [build] rustc 2 <host> -> std 2 <host>
    [build] rustc 2 <host> -> rustc 3 <target1>
    ");
}

#[test]
fn build_compiler_stage_3_full_bootstrap() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("compiler")
            .stage(3)
            .args(&["--set", "build.full-bootstrap=true"])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 2 <host> -> std 2 <host>
    [build] rustc 2 <host> -> rustc 3 <host>
    ");
}

#[test]
fn build_compiler_stage_3_cross_full_bootstrap() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("compiler")
            .stage(3)
            .hosts(&[TEST_TRIPLE_1])
            .args(&["--set", "build.full-bootstrap=true"])
            .render_steps(), @r"
    [build] llvm <host>
    [build] llvm <target1>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 2 <host> -> std 2 <target1>
    [build] rustc 2 <host> -> std 2 <host>
    [build] rustc 2 <host> -> rustc 3 <target1>
    ");
}

#[test]
fn build_compiler_codegen_backend() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx
            .config("build")
            .args(&["--set", "rust.codegen-backends=['llvm', 'cranelift']"])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> rustc_codegen_cranelift 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustdoc 1 <host>
    "
    );
}

#[test]
fn build_compiler_tools() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx
            .config("build")
            .stage(2)
            .args(&["--set", "rust.lld=true", "--set", "rust.llvm-bitcode-linker=true"])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> LldWrapper 1 <host>
    [build] rustc 0 <host> -> LlvmBitcodeLinker 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> LldWrapper 2 <host>
    [build] rustc 1 <host> -> LlvmBitcodeLinker 2 <host>
    [build] rustc 2 <host> -> std 2 <host>
    [build] rustdoc 2 <host>
    "
    );
}

#[test]
fn build_compiler_tools_cross() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx
            .config("build")
            .stage(2)
            .args(&["--set", "rust.lld=true", "--set", "rust.llvm-bitcode-linker=true"])
            .hosts(&[TEST_TRIPLE_1])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> LldWrapper 1 <host>
    [build] rustc 0 <host> -> LlvmBitcodeLinker 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> LldWrapper 2 <host>
    [build] rustc 1 <host> -> LlvmBitcodeLinker 2 <host>
    [build] rustc 1 <host> -> std 1 <target1>
    [build] rustc 2 <host> -> std 2 <target1>
    [build] llvm <target1>
    [build] rustc 1 <host> -> rustc 2 <target1>
    [build] rustc 1 <host> -> LldWrapper 2 <target1>
    [build] rustc 1 <host> -> LlvmBitcodeLinker 2 <target1>
    [build] rustdoc 2 <target1>
    "
    );
}

#[test]
fn build_compiler_lld_opt_in() {
    with_lld_opt_in_targets(vec![host_target()], || {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("compiler")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> LldWrapper 1 <host>
        ");
    });
}

#[test]
fn build_library_no_explicit_stage() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
        .path("library")
        .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    ");
}

#[test]
#[should_panic]
fn build_library_stage_0() {
    let ctx = TestCtx::new();
    ctx.config("build").path("library").stage(0).run();
}

#[test]
fn build_library_stage_0_local_rebuild() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("library")
            .stage(0)
            .targets(&[TEST_TRIPLE_1])
            .args(&["--set", "build.local-rebuild=true"])
            .render_steps(), @"[build] rustc 0 <host> -> std 0 <target1>");
}

#[test]
fn build_library_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("library")
            .stage(1)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    ");
}

#[test]
fn build_library_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("library")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 2 <host> -> std 2 <host>
    ");
}

#[test]
fn build_miri_no_explicit_stage() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("miri")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> miri 1 <host>
    ");
}

#[test]
#[should_panic]
fn build_miri_stage_0() {
    let ctx = TestCtx::new();
    ctx.config("build").path("miri").stage(0).run();
}

#[test]
fn build_miri_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("miri")
            .stage(1)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> miri 1 <host>
    ");
}

#[test]
fn build_miri_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("miri")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> miri 2 <host>
    ");
}

#[test]
fn build_error_index() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("error_index_generator")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> error-index 1 <host>
    ");
}

#[test]
fn build_bootstrap_tool_no_explicit_stage() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("opt-dist")
            .render_steps(), @"[build] rustc 0 <host> -> OptimizedDist 1 <host>");
}

#[test]
#[should_panic]
fn build_bootstrap_tool_stage_0() {
    let ctx = TestCtx::new();
    ctx.config("build").path("opt-dist").stage(0).run();
}

#[test]
fn build_bootstrap_tool_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("opt-dist")
            .stage(1)
            .render_steps(), @"[build] rustc 0 <host> -> OptimizedDist 1 <host>");
}

#[test]
fn build_bootstrap_tool_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .path("opt-dist")
            .stage(2)
            .render_steps(), @"[build] rustc 0 <host> -> OptimizedDist 1 <host>");
}

#[test]
fn build_default_stage() {
    let ctx = TestCtx::new();
    assert_eq!(ctx.config("build").path("compiler").create_config().stage, 1);
}

/// Ensure that if someone passes both a single crate and `library`, all
/// library crates get built.
#[test]
fn alias_and_path_for_library() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(ctx.config("build")
        .paths(&["library", "core"])
        .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    ");

    insta::assert_snapshot!(ctx.config("build")
        .paths(&["std"])
        .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    ");

    insta::assert_snapshot!(ctx.config("build")
        .paths(&["core"])
        .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    ");

    insta::assert_snapshot!(ctx.config("build")
        .paths(&["alloc"])
        .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    ");

    insta::assert_snapshot!(ctx.config("doc")
        .paths(&["library", "core"])
        .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    ");
}

#[test]
fn build_all() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .stage(2)
            .paths(&["compiler/rustc", "library"])
            .hosts(&[&host_target(), TEST_TRIPLE_1])
            .targets(&[&host_target(), TEST_TRIPLE_1, TEST_TRIPLE_2])
        .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] llvm <target1>
    [build] rustc 1 <host> -> std 1 <target1>
    [build] rustc 1 <host> -> rustc 2 <target1>
    [build] rustc 2 <host> -> std 2 <host>
    [build] rustc 2 <host> -> std 2 <target1>
    [build] rustc 1 <host> -> std 1 <target2>
    [build] rustc 2 <host> -> std 2 <target2>
    ");
}

#[test]
fn build_cargo() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .paths(&["cargo"])
        .render_steps(), @"[build] rustc 0 <host> -> cargo 1 <host>");
}

#[test]
fn build_cargo_cross() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("build")
            .paths(&["cargo"])
            .hosts(&[TEST_TRIPLE_1])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> std 1 <target1>
    [build] rustc 1 <host> -> cargo 2 <target1>
    ");
}

#[test]
fn dist_default_stage() {
    let ctx = TestCtx::new();
    assert_eq!(ctx.config("dist").path("compiler").create_config().stage, 2);
}

#[test]
fn dist_baseline() {
    let ctx = TestCtx::new();
    // Note that stdlib is uplifted, that is why `[dist] rustc 1 <host> -> std <host>` is in
    // the output.
    insta::assert_snapshot!(
        ctx
            .config("dist")
            .render_steps(), @r"
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <host>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [doc] book (book) <host>
    [doc] book/first-edition (book) <host>
    [doc] book/second-edition (book) <host>
    [doc] book/2018-edition (book) <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> standalone 2 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> error-index 2 <host>
    [doc] rustc 1 <host> -> error-index 2 <host>
    [doc] nomicon (book) <host>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    [doc] rustdoc (book) <host>
    [doc] rust-by-example (book) <host>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <host>
    [doc] cargo (book) <host>
    [doc] clippy (book) <host>
    [doc] embedded-book (book) <host>
    [doc] edition-guide (book) <host>
    [doc] style-guide (book) <host>
    [doc] rustc 1 <host> -> releases 2 <host>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    [dist] docs <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[]
    [dist] rustc 1 <host> -> json-docs 2 <host>
    [dist] mingw <host>
    [build] rustdoc 2 <host>
    [build] rustc 0 <host> -> GenerateCopyright 1 <host>
    [dist] rustc <host>
    [dist] rustc 1 <host> -> std 1 <host>
    [dist] rustc 1 <host> -> rustc-dev 2 <host>
    [dist] src <>
    [dist] reproducible-artifacts <host>
    "
    );
}

#[test]
fn dist_compiler_docs() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("dist")
            .path("rustc-docs")
            .args(&["--set", "build.compiler-docs=true"])
            .render_steps(), @r"
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <host>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [doc] book (book) <host>
    [doc] book/first-edition (book) <host>
    [doc] book/second-edition (book) <host>
    [doc] book/2018-edition (book) <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> standalone 2 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [doc] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [doc] rustc 1 <host> -> Rustdoc 2 <host>
    [doc] rustc 1 <host> -> Rustfmt 2 <host>
    [build] rustc 1 <host> -> error-index 2 <host>
    [doc] rustc 1 <host> -> error-index 2 <host>
    [doc] nomicon (book) <host>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    [doc] rustdoc (book) <host>
    [doc] rust-by-example (book) <host>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <host>
    [doc] rustc 1 <host> -> Cargo 2 <host>
    [doc] cargo (book) <host>
    [doc] rustc 1 <host> -> Clippy 2 <host>
    [doc] clippy (book) <host>
    [doc] rustc 1 <host> -> Miri 2 <host>
    [doc] embedded-book (book) <host>
    [doc] edition-guide (book) <host>
    [doc] style-guide (book) <host>
    [build] rustdoc 0 <host>
    [doc] rustc 0 <host> -> Tidy 1 <host>
    [doc] rustc 0 <host> -> Bootstrap 1 <host>
    [doc] rustc 1 <host> -> releases 2 <host>
    [doc] rustc 0 <host> -> RunMakeSupport 1 <host>
    [doc] rustc 0 <host> -> BuildHelper 1 <host>
    [doc] rustc 0 <host> -> Compiletest 1 <host>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    "
    );
}

#[test]
fn dist_extended() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("dist")
        .args(&[
            "--set",
            "build.extended=true",
            "--set",
            "rust.llvm-bitcode-linker=true",
            "--set",
            "rust.lld=true",
        ])
        .render_steps(), @r"
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <host>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> LldWrapper 1 <host>
    [build] rustc 0 <host> -> WasmComponentLd 1 <host>
    [build] rustc 0 <host> -> LlvmBitcodeLinker 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [doc] book (book) <host>
    [doc] book/first-edition (book) <host>
    [doc] book/second-edition (book) <host>
    [doc] book/2018-edition (book) <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> standalone 2 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> LldWrapper 2 <host>
    [build] rustc 1 <host> -> WasmComponentLd 2 <host>
    [build] rustc 1 <host> -> LlvmBitcodeLinker 2 <host>
    [build] rustc 1 <host> -> error-index 2 <host>
    [doc] rustc 1 <host> -> error-index 2 <host>
    [doc] nomicon (book) <host>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    [doc] rustdoc (book) <host>
    [doc] rust-by-example (book) <host>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <host>
    [doc] cargo (book) <host>
    [doc] clippy (book) <host>
    [doc] embedded-book (book) <host>
    [doc] edition-guide (book) <host>
    [doc] style-guide (book) <host>
    [doc] rustc 1 <host> -> releases 2 <host>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    [dist] docs <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[]
    [dist] rustc 1 <host> -> json-docs 2 <host>
    [dist] mingw <host>
    [build] rustdoc 2 <host>
    [build] rustc 1 <host> -> rust-analyzer-proc-macro-srv 2 <host>
    [build] rustc 0 <host> -> GenerateCopyright 1 <host>
    [dist] rustc <host>
    [dist] rustc 1 <host> -> std 1 <host>
    [dist] rustc 1 <host> -> rustc-dev 2 <host>
    [dist] rustc 1 <host> -> analysis 2 <host>
    [dist] src <>
    [build] rustc 1 <host> -> cargo 2 <host>
    [dist] rustc 1 <host> -> cargo 2 <host>
    [build] rustc 1 <host> -> rust-analyzer 2 <host>
    [dist] rustc 1 <host> -> rust-analyzer 2 <host>
    [build] rustc 1 <host> -> rustfmt 2 <host>
    [build] rustc 1 <host> -> cargo-fmt 2 <host>
    [dist] rustc 1 <host> -> rustfmt 2 <host>
    [build] rustc 1 <host> -> clippy-driver 2 <host>
    [build] rustc 1 <host> -> cargo-clippy 2 <host>
    [dist] rustc 1 <host> -> clippy 2 <host>
    [build] rustc 1 <host> -> miri 2 <host>
    [build] rustc 1 <host> -> cargo-miri 2 <host>
    [dist] rustc 1 <host> -> miri 2 <host>
    [doc] rustc 2 <host> -> std 2 <host> crates=[]
    [dist] rustc 2 <host> -> json-docs 3 <host>
    [dist] rustc 1 <host> -> extended 2 <host>
    [dist] reproducible-artifacts <host>
    ");
}

#[test]
fn dist_with_targets() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx
            .config("dist")
            .hosts(&[&host_target()])
            .targets(&[&host_target(), TEST_TRIPLE_1])
            .render_steps(), @r"
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <host>
    [doc] unstable-book (book) <target1>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [doc] book (book) <host>
    [doc] book/first-edition (book) <host>
    [doc] book/second-edition (book) <host>
    [doc] book/2018-edition (book) <host>
    [build] rustdoc 1 <host>
    [build] rustc 1 <host> -> std 1 <target1>
    [doc] book (book) <target1>
    [doc] book/first-edition (book) <target1>
    [doc] book/second-edition (book) <target1>
    [doc] book/2018-edition (book) <target1>
    [doc] rustc 1 <host> -> standalone 2 <host>
    [doc] rustc 1 <host> -> standalone 2 <target1>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [doc] rustc 1 <host> -> std 1 <target1> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> error-index 2 <host>
    [doc] rustc 1 <host> -> error-index 2 <host>
    [doc] nomicon (book) <host>
    [doc] nomicon (book) <target1>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    [doc] rustc 1 <host> -> reference (book) 2 <target1>
    [doc] rustdoc (book) <host>
    [doc] rustdoc (book) <target1>
    [doc] rust-by-example (book) <host>
    [doc] rust-by-example (book) <target1>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <host>
    [doc] cargo (book) <host>
    [doc] cargo (book) <target1>
    [doc] clippy (book) <host>
    [doc] clippy (book) <target1>
    [doc] embedded-book (book) <host>
    [doc] embedded-book (book) <target1>
    [doc] edition-guide (book) <host>
    [doc] edition-guide (book) <target1>
    [doc] style-guide (book) <host>
    [doc] style-guide (book) <target1>
    [doc] rustc 1 <host> -> releases 2 <host>
    [doc] rustc 1 <host> -> releases 2 <target1>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    [dist] docs <host>
    [dist] docs <target1>
    [doc] rustc 1 <host> -> std 1 <host> crates=[]
    [dist] rustc 1 <host> -> json-docs 2 <host>
    [doc] rustc 1 <host> -> std 1 <target1> crates=[]
    [dist] rustc 1 <host> -> json-docs 2 <target1>
    [dist] mingw <host>
    [dist] mingw <target1>
    [build] rustdoc 2 <host>
    [build] rustc 0 <host> -> GenerateCopyright 1 <host>
    [dist] rustc <host>
    [dist] rustc 1 <host> -> std 1 <host>
    [dist] rustc 1 <host> -> std 1 <target1>
    [dist] rustc 1 <host> -> rustc-dev 2 <host>
    [dist] src <>
    [dist] reproducible-artifacts <host>
    "
    );
}

#[test]
fn dist_with_hosts() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx
            .config("dist")
            .hosts(&[&host_target(), TEST_TRIPLE_1])
            .targets(&[&host_target()])
            .render_steps(), @r"
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <host>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [doc] book (book) <host>
    [doc] book/first-edition (book) <host>
    [doc] book/second-edition (book) <host>
    [doc] book/2018-edition (book) <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> standalone 2 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> error-index 2 <host>
    [doc] rustc 1 <host> -> error-index 2 <host>
    [build] llvm <target1>
    [build] rustc 1 <host> -> std 1 <target1>
    [build] rustc 1 <host> -> rustc 2 <target1>
    [build] rustc 1 <host> -> error-index 2 <target1>
    [doc] rustc 1 <host> -> error-index 2 <target1>
    [doc] nomicon (book) <host>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    [doc] rustdoc (book) <host>
    [doc] rust-by-example (book) <host>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <host>
    [doc] rustc (book) <target1>
    [doc] cargo (book) <host>
    [doc] clippy (book) <host>
    [doc] embedded-book (book) <host>
    [doc] edition-guide (book) <host>
    [doc] style-guide (book) <host>
    [doc] rustc 1 <host> -> releases 2 <host>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    [dist] docs <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[]
    [dist] rustc 1 <host> -> json-docs 2 <host>
    [dist] mingw <host>
    [build] rustdoc 2 <host>
    [build] rustc 0 <host> -> GenerateCopyright 1 <host>
    [dist] rustc <host>
    [build] rustdoc 2 <target1>
    [dist] rustc <target1>
    [dist] rustc 1 <host> -> std 1 <host>
    [dist] rustc 1 <host> -> rustc-dev 2 <host>
    [dist] rustc 1 <host> -> rustc-dev 2 <target1>
    [dist] src <>
    [dist] reproducible-artifacts <host>
    [dist] reproducible-artifacts <target1>
    "
    );
}

#[test]
fn dist_with_targets_and_hosts() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx
            .config("dist")
            .hosts(&[&host_target(), TEST_TRIPLE_1])
            .targets(&[&host_target(), TEST_TRIPLE_1])
            .render_steps(), @r"
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <host>
    [doc] unstable-book (book) <target1>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [doc] book (book) <host>
    [doc] book/first-edition (book) <host>
    [doc] book/second-edition (book) <host>
    [doc] book/2018-edition (book) <host>
    [build] rustdoc 1 <host>
    [build] rustc 1 <host> -> std 1 <target1>
    [doc] book (book) <target1>
    [doc] book/first-edition (book) <target1>
    [doc] book/second-edition (book) <target1>
    [doc] book/2018-edition (book) <target1>
    [doc] rustc 1 <host> -> standalone 2 <host>
    [doc] rustc 1 <host> -> standalone 2 <target1>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [doc] rustc 1 <host> -> std 1 <target1> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> error-index 2 <host>
    [doc] rustc 1 <host> -> error-index 2 <host>
    [build] llvm <target1>
    [build] rustc 1 <host> -> rustc 2 <target1>
    [build] rustc 1 <host> -> error-index 2 <target1>
    [doc] rustc 1 <host> -> error-index 2 <target1>
    [doc] nomicon (book) <host>
    [doc] nomicon (book) <target1>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    [doc] rustc 1 <host> -> reference (book) 2 <target1>
    [doc] rustdoc (book) <host>
    [doc] rustdoc (book) <target1>
    [doc] rust-by-example (book) <host>
    [doc] rust-by-example (book) <target1>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <host>
    [doc] rustc (book) <target1>
    [doc] cargo (book) <host>
    [doc] cargo (book) <target1>
    [doc] clippy (book) <host>
    [doc] clippy (book) <target1>
    [doc] embedded-book (book) <host>
    [doc] embedded-book (book) <target1>
    [doc] edition-guide (book) <host>
    [doc] edition-guide (book) <target1>
    [doc] style-guide (book) <host>
    [doc] style-guide (book) <target1>
    [doc] rustc 1 <host> -> releases 2 <host>
    [doc] rustc 1 <host> -> releases 2 <target1>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    [dist] docs <host>
    [dist] docs <target1>
    [doc] rustc 1 <host> -> std 1 <host> crates=[]
    [dist] rustc 1 <host> -> json-docs 2 <host>
    [doc] rustc 1 <host> -> std 1 <target1> crates=[]
    [dist] rustc 1 <host> -> json-docs 2 <target1>
    [dist] mingw <host>
    [dist] mingw <target1>
    [build] rustdoc 2 <host>
    [build] rustc 0 <host> -> GenerateCopyright 1 <host>
    [dist] rustc <host>
    [build] rustdoc 2 <target1>
    [dist] rustc <target1>
    [dist] rustc 1 <host> -> std 1 <host>
    [dist] rustc 1 <host> -> std 1 <target1>
    [dist] rustc 1 <host> -> rustc-dev 2 <host>
    [dist] rustc 1 <host> -> rustc-dev 2 <target1>
    [dist] src <>
    [dist] reproducible-artifacts <host>
    [dist] reproducible-artifacts <target1>
    "
    );
}

#[test]
fn dist_with_empty_host() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx
            .config("dist")
            .hosts(&[])
            .targets(&[TEST_TRIPLE_1])
            .render_steps(), @r"
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <target1>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <target1>
    [doc] book (book) <target1>
    [doc] book/first-edition (book) <target1>
    [doc] book/second-edition (book) <target1>
    [doc] book/2018-edition (book) <target1>
    [build] rustdoc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [doc] rustc 1 <host> -> standalone 2 <target1>
    [doc] rustc 1 <host> -> std 1 <target1> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [doc] nomicon (book) <target1>
    [doc] rustc 1 <host> -> reference (book) 2 <target1>
    [doc] rustdoc (book) <target1>
    [doc] rust-by-example (book) <target1>
    [doc] cargo (book) <target1>
    [doc] clippy (book) <target1>
    [doc] embedded-book (book) <target1>
    [doc] edition-guide (book) <target1>
    [doc] style-guide (book) <target1>
    [doc] rustc 1 <host> -> releases 2 <target1>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    [dist] docs <target1>
    [doc] rustc 1 <host> -> std 1 <target1> crates=[]
    [dist] rustc 1 <host> -> json-docs 2 <target1>
    [dist] mingw <target1>
    [dist] rustc 1 <host> -> std 1 <target1>
    ");
}

#[test]
fn dist_all_cross_extended() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx
            .config("dist")
            .hosts(&[TEST_TRIPLE_1])
            .targets(&[TEST_TRIPLE_1])
            .args(&["--set", "rust.channel=nightly", "--set", "build.extended=true"])
            .render_steps(), @r"
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <target1>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> WasmComponentLd 1 <host>
    [build] rustc 1 <host> -> std 1 <target1>
    [doc] book (book) <target1>
    [doc] book/first-edition (book) <target1>
    [doc] book/second-edition (book) <target1>
    [doc] book/2018-edition (book) <target1>
    [build] rustdoc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [doc] rustc 1 <host> -> standalone 2 <target1>
    [doc] rustc 1 <host> -> std 1 <target1> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [build] llvm <target1>
    [build] rustc 1 <host> -> rustc 2 <target1>
    [build] rustc 1 <host> -> WasmComponentLd 2 <target1>
    [build] rustc 1 <host> -> error-index 2 <target1>
    [doc] rustc 1 <host> -> error-index 2 <target1>
    [doc] nomicon (book) <target1>
    [doc] rustc 1 <host> -> reference (book) 2 <target1>
    [doc] rustdoc (book) <target1>
    [doc] rust-by-example (book) <target1>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <target1>
    [doc] cargo (book) <target1>
    [doc] clippy (book) <target1>
    [doc] embedded-book (book) <target1>
    [doc] edition-guide (book) <target1>
    [doc] style-guide (book) <target1>
    [doc] rustc 1 <host> -> releases 2 <target1>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    [dist] docs <target1>
    [doc] rustc 1 <host> -> std 1 <target1> crates=[]
    [dist] rustc 1 <host> -> json-docs 2 <target1>
    [dist] mingw <target1>
    [build] rustdoc 2 <target1>
    [build] rustc 1 <host> -> rust-analyzer-proc-macro-srv 2 <target1>
    [build] rustc 0 <host> -> GenerateCopyright 1 <host>
    [dist] rustc <target1>
    [dist] rustc 1 <host> -> std 1 <target1>
    [dist] rustc 1 <host> -> rustc-dev 2 <target1>
    [dist] rustc 1 <host> -> analysis 2 <target1>
    [dist] src <>
    [build] rustc 1 <host> -> cargo 2 <target1>
    [dist] rustc 1 <host> -> cargo 2 <target1>
    [build] rustc 1 <host> -> rust-analyzer 2 <target1>
    [dist] rustc 1 <host> -> rust-analyzer 2 <target1>
    [build] rustc 1 <host> -> rustfmt 2 <target1>
    [build] rustc 1 <host> -> cargo-fmt 2 <target1>
    [dist] rustc 1 <host> -> rustfmt 2 <target1>
    [build] rustc 1 <host> -> clippy-driver 2 <target1>
    [build] rustc 1 <host> -> cargo-clippy 2 <target1>
    [dist] rustc 1 <host> -> clippy 2 <target1>
    [build] rustc 1 <host> -> miri 2 <target1>
    [build] rustc 1 <host> -> cargo-miri 2 <target1>
    [dist] rustc 1 <host> -> miri 2 <target1>
    [build] rustc 1 <host> -> LlvmBitcodeLinker 2 <target1>
    [doc] rustc 2 <target1> -> std 2 <target1> crates=[]
    [dist] rustc 2 <target1> -> json-docs 3 <target1>
    [dist] rustc 1 <host> -> extended 2 <target1>
    [dist] reproducible-artifacts <target1>
    ");
}

/// Simulates e.g. the powerpc64 builder, which is fully cross-compiled from x64, but it does
/// not build docs. Crucially, it shouldn't build host stage 2 rustc.
///
/// This is a regression test for <https://github.com/rust-lang/rust/issues/138123>
/// and <https://github.com/rust-lang/rust/issues/138004>.
#[test]
fn dist_all_cross_extended_no_docs() {
    let ctx = TestCtx::new();
    let steps = ctx
        .config("dist")
        .hosts(&[TEST_TRIPLE_1])
        .targets(&[TEST_TRIPLE_1])
        .args(&[
            "--set",
            "rust.channel=nightly",
            "--set",
            "build.extended=true",
            "--set",
            "build.docs=false",
        ])
        .get_steps();

    // Make sure that we don't build stage2 host rustc
    steps.assert_no_match(|m| {
        m.name == "rustc"
            && m.built_by.map(|b| b.stage) == Some(1)
            && *m.target.triple == host_target()
    });

    insta::assert_snapshot!(
            steps.render(), @r"
    [dist] mingw <target1>
    [build] llvm <host>
    [build] llvm <target1>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> WasmComponentLd 1 <host>
    [build] rustc 1 <host> -> std 1 <target1>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <target1>
    [build] rustc 1 <host> -> WasmComponentLd 2 <target1>
    [build] rustdoc 2 <target1>
    [build] rustc 1 <host> -> rust-analyzer-proc-macro-srv 2 <target1>
    [build] rustc 0 <host> -> GenerateCopyright 1 <host>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    [dist] rustc <target1>
    [dist] rustc 1 <host> -> std 1 <target1>
    [dist] rustc 1 <host> -> rustc-dev 2 <target1>
    [dist] rustc 1 <host> -> analysis 2 <target1>
    [dist] src <>
    [build] rustc 1 <host> -> cargo 2 <target1>
    [dist] rustc 1 <host> -> cargo 2 <target1>
    [build] rustc 1 <host> -> rust-analyzer 2 <target1>
    [dist] rustc 1 <host> -> rust-analyzer 2 <target1>
    [build] rustc 1 <host> -> rustfmt 2 <target1>
    [build] rustc 1 <host> -> cargo-fmt 2 <target1>
    [dist] rustc 1 <host> -> rustfmt 2 <target1>
    [build] rustc 1 <host> -> clippy-driver 2 <target1>
    [build] rustc 1 <host> -> cargo-clippy 2 <target1>
    [dist] rustc 1 <host> -> clippy 2 <target1>
    [build] rustc 1 <host> -> miri 2 <target1>
    [build] rustc 1 <host> -> cargo-miri 2 <target1>
    [dist] rustc 1 <host> -> miri 2 <target1>
    [build] rustc 1 <host> -> LlvmBitcodeLinker 2 <target1>
    [dist] rustc 1 <host> -> extended 2 <target1>
    [dist] reproducible-artifacts <target1>
    ");
}

/// Enable dist cranelift tarball by default with `x dist` if cranelift is enabled in
/// `rust.codegen-backends`.
#[test]
fn dist_cranelift_by_default() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx
            .config("dist")
            .args(&["--set", "rust.codegen-backends=['llvm', 'cranelift']"])
            .render_steps(), @r"
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <host>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> rustc_codegen_cranelift 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [doc] book (book) <host>
    [doc] book/first-edition (book) <host>
    [doc] book/second-edition (book) <host>
    [doc] book/2018-edition (book) <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> standalone 2 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> rustc_codegen_cranelift 2 <host>
    [build] rustc 1 <host> -> error-index 2 <host>
    [doc] rustc 1 <host> -> error-index 2 <host>
    [doc] nomicon (book) <host>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    [doc] rustdoc (book) <host>
    [doc] rust-by-example (book) <host>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <host>
    [doc] cargo (book) <host>
    [doc] clippy (book) <host>
    [doc] embedded-book (book) <host>
    [doc] edition-guide (book) <host>
    [doc] style-guide (book) <host>
    [doc] rustc 1 <host> -> releases 2 <host>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    [dist] docs <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[]
    [dist] rustc 1 <host> -> json-docs 2 <host>
    [dist] mingw <host>
    [build] rustdoc 2 <host>
    [build] rustc 0 <host> -> GenerateCopyright 1 <host>
    [dist] rustc <host>
    [dist] rustc 1 <host> -> rustc_codegen_cranelift 2 <host>
    [dist] rustc 1 <host> -> std 1 <host>
    [dist] rustc 1 <host> -> rustc-dev 2 <host>
    [dist] src <>
    [dist] reproducible-artifacts <host>
    ");
}

#[test]
fn dist_bootstrap() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx
            .config("dist")
            .path("bootstrap")
            .render_steps(), @r"
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    [dist] bootstrap <host>
    ");
}

#[test]
fn dist_library_stage_0_local_rebuild() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("dist")
            .path("rust-std")
            .stage(0)
            .targets(&[TEST_TRIPLE_1])
            .args(&["--set", "build.local-rebuild=true"])
            .render_steps(), @r"
    [build] rustc 0 <host> -> std 0 <target1>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    [dist] rustc 0 <host> -> std 0 <target1>
    ");
}

#[test]
fn dist_rustc_docs() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx
            .config("dist")
            .path("rustc-docs")
            .render_steps(), @r"
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <host>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [doc] book (book) <host>
    [doc] book/first-edition (book) <host>
    [doc] book/second-edition (book) <host>
    [doc] book/2018-edition (book) <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> standalone 2 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> error-index 2 <host>
    [doc] rustc 1 <host> -> error-index 2 <host>
    [doc] nomicon (book) <host>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    [doc] rustdoc (book) <host>
    [doc] rust-by-example (book) <host>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <host>
    [doc] cargo (book) <host>
    [doc] clippy (book) <host>
    [doc] embedded-book (book) <host>
    [doc] edition-guide (book) <host>
    [doc] style-guide (book) <host>
    [doc] rustc 1 <host> -> releases 2 <host>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    ");
}

#[test]
fn check_compiler_no_explicit_stage() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("compiler")
            .render_steps(), @"[check] rustc 0 <host> -> rustc 1 <host> (75 crates)");
}

#[test]
fn check_rustc_no_explicit_stage() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("rustc")
            .render_steps(), @"[check] rustc 0 <host> -> rustc 1 <host> (1 crates)");
}

#[test]
#[should_panic]
fn check_compiler_stage_0() {
    let ctx = TestCtx::new();
    ctx.config("check").path("compiler").stage(0).run();
}

#[test]
fn check_compiler_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("compiler")
            .stage(1)
            .render_steps(), @"[check] rustc 0 <host> -> rustc 1 <host> (75 crates)");
}

#[test]
fn check_compiler_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("compiler")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [check] rustc 1 <host> -> rustc 2 <host> (75 crates)
    ");
}

#[test]
fn check_cross_compile() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .targets(&[TEST_TRIPLE_1])
            .hosts(&[TEST_TRIPLE_1])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [check] rustc 1 <host> -> std 1 <target1>
    [check] rustc 1 <host> -> rustc 2 <target1> (75 crates)
    [check] rustc 1 <host> -> rustc 2 <target1>
    [check] rustc 1 <host> -> Rustdoc 2 <target1>
    [check] rustc 1 <host> -> rustc_codegen_cranelift 2 <target1>
    [check] rustc 1 <host> -> rustc_codegen_gcc 2 <target1>
    [check] rustc 1 <host> -> Clippy 2 <target1>
    [check] rustc 1 <host> -> Miri 2 <target1>
    [check] rustc 1 <host> -> CargoMiri 2 <target1>
    [check] rustc 1 <host> -> Rustfmt 2 <target1>
    [check] rustc 1 <host> -> RustAnalyzer 2 <target1>
    [check] rustc 1 <host> -> TestFloatParse 2 <target1>
    [check] rustc 1 <host> -> std 1 <target1>
    ");
}

#[test]
fn check_library_no_explicit_stage() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("library")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [check] rustc 1 <host> -> std 1 <host>
    ");
}

#[test]
#[should_panic]
fn check_library_stage_0() {
    let ctx = TestCtx::new();
    ctx.config("check").path("library").stage(0).run();
}

#[test]
fn check_library_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("library")
            .stage(1)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [check] rustc 1 <host> -> std 1 <host>
    ");
}

#[test]
fn check_library_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("library")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [check] rustc 2 <host> -> std 2 <host>
    ");
}

#[test]
fn check_library_cross_compile() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .paths(&["core", "alloc", "std"])
            .targets(&[TEST_TRIPLE_1, TEST_TRIPLE_2])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [check] rustc 1 <host> -> std 1 <target1>
    [check] rustc 1 <host> -> std 1 <target2>
    ");
}

/// Make sure that we don't check library when download-rustc is disabled
/// when `--skip-std-check-if-no-download-rustc` was passed.
#[test]
fn check_library_skip_without_download_rustc() {
    let ctx = TestCtx::new();
    let args = ["--set", "rust.download-rustc=false", "--skip-std-check-if-no-download-rustc"];
    insta::assert_snapshot!(
        ctx.config("check")
            .paths(&["library"])
            .args(&args)
            .render_steps(), @"");

    insta::assert_snapshot!(
        ctx.config("check")
            .paths(&["library", "compiler"])
            .args(&args)
            .render_steps(), @"[check] rustc 0 <host> -> rustc 1 <host> (75 crates)");
}

#[test]
fn check_miri_no_explicit_stage() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("miri")
            .render_steps(), @r"
    [check] rustc 0 <host> -> rustc 1 <host>
    [check] rustc 0 <host> -> Miri 1 <host>
    ");
}

#[test]
#[should_panic]
fn check_miri_stage_0() {
    let ctx = TestCtx::new();
    ctx.config("check").path("miri").stage(0).run();
}

#[test]
fn check_miri_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("miri")
            .stage(1)
            .render_steps(), @r"
    [check] rustc 0 <host> -> rustc 1 <host>
    [check] rustc 0 <host> -> Miri 1 <host>
    ");
}

#[test]
fn check_miri_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("miri")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [check] rustc 1 <host> -> rustc 2 <host>
    [check] rustc 1 <host> -> Miri 2 <host>
    ");
}

#[test]
fn check_compiletest() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("compiletest")
            .render_steps(), @"[check] rustc 0 <host> -> Compiletest 1 <host>");
}

#[test]
fn check_compiletest_stage1_libtest() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("compiletest")
            .args(&["--set", "build.compiletest-use-stage0-libtest=false"])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [check] rustc 1 <host> -> Compiletest 2 <host>
    ");
}

#[test]
fn check_codegen() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("rustc_codegen_cranelift")
            .render_steps(), @r"
    [check] rustc 0 <host> -> rustc 1 <host>
    [check] rustc 0 <host> -> rustc_codegen_cranelift 1 <host>
    ");
}

#[test]
fn check_rust_analyzer() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("rust-analyzer")
            .render_steps(), @r"
    [check] rustc 0 <host> -> rustc 1 <host>
    [check] rustc 0 <host> -> RustAnalyzer 1 <host>
    ");
}

#[test]
fn check_bootstrap_tool() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("check")
            .path("run-make-support")
            .render_steps(), @"[check] rustc 0 <host> -> RunMakeSupport 1 <host>");
}

fn prepare_test_config(ctx: &TestCtx) -> ConfigBuilder {
    ctx.config("test")
        // Bootstrap only runs by default on CI, so we have to emulate that also locally.
        .args(&["--ci", "true"])
        // These rustdoc tests requires nodejs to be present.
        // We can't easily opt out of it, so if it is present on the local PC, the test
        // would have different result on CI, where nodejs might be missing.
        .args(&["--skip", "rustdoc-js-std"])
        .args(&["--skip", "rustdoc-js"])
        .args(&["--skip", "rustdoc-gui"])
}

#[test]
fn test_all_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        prepare_test_config(&ctx)
            .render_steps(), @r"
    [build] rustc 0 <host> -> Tidy 1 <host>
    [test] tidy <>
    [build] rustdoc 0 <host>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 0 <host> -> Compiletest 1 <host>
    [test] compiletest-ui 1 <host>
    [test] compiletest-crashes 1 <host>
    [build] rustc 0 <host> -> CoverageDump 1 <host>
    [test] compiletest-coverage 1 <host>
    [test] compiletest-coverage 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [test] compiletest-mir-opt 1 <host>
    [test] compiletest-codegen-llvm 1 <host>
    [test] compiletest-codegen-units 1 <host>
    [test] compiletest-assembly-llvm 1 <host>
    [test] compiletest-incremental 1 <host>
    [test] compiletest-debuginfo 1 <host>
    [test] compiletest-ui-fulldeps 1 <host>
    [build] rustdoc 1 <host>
    [test] compiletest-rustdoc 1 <host>
    [test] compiletest-coverage-run-rustdoc 1 <host>
    [test] compiletest-pretty 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 0 <host> -> std 0 <host>
    [test] rustc 0 <host> -> CrateLibrustc 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [test] crate-bootstrap <host> src/tools/coverage-dump
    [test] crate-bootstrap <host> src/tools/jsondoclint
    [test] crate-bootstrap <host> src/tools/replace-version-placeholder
    [test] crate-bootstrap <host> tidyselftest
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <host>
    [doc] book (book) <host>
    [doc] book/first-edition (book) <host>
    [doc] book/second-edition (book) <host>
    [doc] book/2018-edition (book) <host>
    [doc] rustc 0 <host> -> standalone 1 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [build] rustc 0 <host> -> error-index 1 <host>
    [doc] rustc 0 <host> -> error-index 1 <host>
    [doc] nomicon (book) <host>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    [doc] rustdoc (book) <host>
    [doc] rust-by-example (book) <host>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <host>
    [doc] cargo (book) <host>
    [doc] clippy (book) <host>
    [doc] embedded-book (book) <host>
    [doc] edition-guide (book) <host>
    [doc] style-guide (book) <host>
    [doc] rustc 0 <host> -> releases 1 <host>
    [build] rustc 0 <host> -> Linkchecker 1 <host>
    [test] link-check <host>
    [test] tier-check <host>
    [test] rustc 0 <host> -> rust-analyzer 1 <host>
    [build] rustc 0 <host> -> RustdocTheme 1 <host>
    [test] rustdoc-theme 1 <host>
    [test] compiletest-rustdoc-ui 1 <host>
    [build] rustc 0 <host> -> JsonDocCk 1 <host>
    [build] rustc 0 <host> -> JsonDocLint 1 <host>
    [test] compiletest-rustdoc-json 1 <host>
    [doc] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> HtmlChecker 1 <host>
    [test] html-check <host>
    [build] rustc 0 <host> -> RunMakeSupport 1 <host>
    [test] compiletest-run-make 1 <host>
    [build] rustc 0 <host> -> cargo 1 <host>
    [test] compiletest-run-make-cargo 1 <host>
    ");
}

#[test]
fn test_compiletest_suites_stage1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("test")
            .args(&["ui", "ui-fulldeps", "run-make", "rustdoc", "rustdoc-gui", "incremental"])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 0 <host> -> Compiletest 1 <host>
    [test] compiletest-ui 1 <host>
    [test] compiletest-ui-fulldeps 1 <host>
    [build] rustc 0 <host> -> RunMakeSupport 1 <host>
    [build] rustdoc 1 <host>
    [test] compiletest-run-make 1 <host>
    [test] compiletest-rustdoc 1 <host>
    [build] rustc 0 <host> -> RustdocGUITest 1 <host>
    [test] rustdoc-gui 1 <host>
    [test] compiletest-incremental 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    ");
}

#[test]
fn test_compiletest_suites_stage2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("test")
            .args(&["ui", "ui-fulldeps", "run-make", "rustdoc", "rustdoc-gui", "incremental"])
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 2 <host> -> std 2 <host>
    [build] rustc 0 <host> -> Compiletest 1 <host>
    [test] compiletest-ui 2 <host>
    [build] rustc 2 <host> -> rustc 3 <host>
    [test] compiletest-ui-fulldeps 2 <host>
    [build] rustc 0 <host> -> RunMakeSupport 1 <host>
    [build] rustdoc 2 <host>
    [test] compiletest-run-make 2 <host>
    [test] compiletest-rustdoc 2 <host>
    [build] rustc 0 <host> -> RustdocGUITest 1 <host>
    [test] rustdoc-gui 2 <host>
    [test] compiletest-incremental 2 <host>
    [build] rustdoc 1 <host>
    ");
}

#[test]
fn test_compiletest_suites_stage2_cross() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("test")
            .hosts(&[TEST_TRIPLE_1])
            .targets(&[TEST_TRIPLE_1])
            .args(&["ui", "ui-fulldeps", "run-make", "rustdoc", "rustdoc-gui", "incremental"])
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 2 <host> -> std 2 <host>
    [build] rustc 0 <host> -> Compiletest 1 <host>
    [build] rustc 1 <host> -> std 1 <target1>
    [build] rustc 2 <host> -> std 2 <target1>
    [test] compiletest-ui 2 <target1>
    [build] llvm <target1>
    [build] rustc 2 <host> -> rustc 3 <target1>
    [test] compiletest-ui-fulldeps 2 <target1>
    [build] rustc 0 <host> -> RunMakeSupport 1 <host>
    [build] rustdoc 2 <host>
    [test] compiletest-run-make 2 <target1>
    [test] compiletest-rustdoc 2 <target1>
    [build] rustc 0 <host> -> RustdocGUITest 1 <host>
    [test] rustdoc-gui 2 <target1>
    [test] compiletest-incremental 2 <target1>
    [build] rustc 1 <host> -> rustc 2 <target1>
    [build] rustdoc 1 <host>
    [build] rustc 2 <target1> -> std 2 <target1>
    [build] rustdoc 2 <target1>
    ");
}

#[test]
fn test_all_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        prepare_test_config(&ctx)
            .stage(2)
            .render_steps(), @r"
    [build] rustc 0 <host> -> Tidy 1 <host>
    [test] tidy <>
    [build] rustdoc 0 <host>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 2 <host> -> std 2 <host>
    [build] rustc 0 <host> -> Compiletest 1 <host>
    [test] compiletest-ui 2 <host>
    [test] compiletest-crashes 2 <host>
    [build] rustc 0 <host> -> CoverageDump 1 <host>
    [test] compiletest-coverage 2 <host>
    [test] compiletest-coverage 2 <host>
    [build] rustc 2 <host> -> std 2 <host>
    [test] compiletest-mir-opt 2 <host>
    [test] compiletest-codegen-llvm 2 <host>
    [test] compiletest-codegen-units 2 <host>
    [test] compiletest-assembly-llvm 2 <host>
    [test] compiletest-incremental 2 <host>
    [test] compiletest-debuginfo 2 <host>
    [build] rustc 2 <host> -> rustc 3 <host>
    [test] compiletest-ui-fulldeps 2 <host>
    [build] rustdoc 2 <host>
    [test] compiletest-rustdoc 2 <host>
    [test] compiletest-coverage-run-rustdoc 2 <host>
    [test] compiletest-pretty 2 <host>
    [build] rustc 2 <host> -> std 2 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustdoc 1 <host>
    [test] rustc 1 <host> -> CrateLibrustc 2 <host>
    [test] crate-bootstrap <host> src/tools/coverage-dump
    [test] crate-bootstrap <host> src/tools/jsondoclint
    [test] crate-bootstrap <host> src/tools/replace-version-placeholder
    [test] crate-bootstrap <host> tidyselftest
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <host>
    [doc] book (book) <host>
    [doc] book/first-edition (book) <host>
    [doc] book/second-edition (book) <host>
    [doc] book/2018-edition (book) <host>
    [doc] rustc 1 <host> -> standalone 2 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [build] rustc 1 <host> -> error-index 2 <host>
    [doc] rustc 1 <host> -> error-index 2 <host>
    [doc] nomicon (book) <host>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    [doc] rustdoc (book) <host>
    [doc] rust-by-example (book) <host>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <host>
    [doc] cargo (book) <host>
    [doc] clippy (book) <host>
    [doc] embedded-book (book) <host>
    [doc] edition-guide (book) <host>
    [doc] style-guide (book) <host>
    [doc] rustc 1 <host> -> releases 2 <host>
    [build] rustc 0 <host> -> Linkchecker 1 <host>
    [test] link-check <host>
    [test] tier-check <host>
    [test] rustc 1 <host> -> rust-analyzer 2 <host>
    [doc] rustc (book) <host>
    [test] rustc 1 <host> -> lint-docs 2 <host>
    [build] rustc 0 <host> -> RustdocTheme 1 <host>
    [test] rustdoc-theme 2 <host>
    [test] compiletest-rustdoc-ui 2 <host>
    [build] rustc 0 <host> -> JsonDocCk 1 <host>
    [build] rustc 0 <host> -> JsonDocLint 1 <host>
    [test] compiletest-rustdoc-json 2 <host>
    [doc] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 0 <host> -> HtmlChecker 1 <host>
    [test] html-check <host>
    [build] rustc 0 <host> -> RunMakeSupport 1 <host>
    [test] compiletest-run-make 2 <host>
    [build] rustc 1 <host> -> cargo 2 <host>
    [test] compiletest-run-make-cargo 2 <host>
    ");
}

#[test]
fn test_compiler_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("test")
            .path("compiler")
            .stage(1)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> std 0 <host>
    [build] rustdoc 0 <host>
    [test] rustc 0 <host> -> CrateLibrustc 1 <host>
    ");
}

#[test]
fn test_compiler_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("test")
            .path("compiler")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustdoc 1 <host>
    [test] rustc 1 <host> -> CrateLibrustc 2 <host>
    ");
}

#[test]
fn test_exclude() {
    let ctx = TestCtx::new();
    let steps = ctx.config("test").args(&["--skip", "src/tools/tidy"]).get_steps();

    let host = TargetSelection::from_user(&host_target());
    steps.assert_contains(StepMetadata::test("compiletest-rustdoc-ui", host).stage(1));
    steps.assert_not_contains(test::Tidy);
}

#[test]
fn test_exclude_kind() {
    let ctx = TestCtx::new();
    let host = TargetSelection::from_user(&host_target());

    let get_steps = |args: &[&str]| ctx.config("test").args(args).get_steps();

    let rustc_metadata =
        || StepMetadata::test("CrateLibrustc", host).built_by(Compiler::new(0, host));
    // Ensure our test is valid, and `test::Rustc` would be run without the exclude.
    get_steps(&[]).assert_contains(rustc_metadata());

    let steps = get_steps(&["--skip", "compiler/rustc_data_structures"]);

    // Ensure tests for rustc are not skipped.
    steps.assert_contains(rustc_metadata());
    steps.assert_contains_fuzzy(StepMetadata::build("rustc", host));
}

#[test]
fn test_cargo_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("test")
            .path("cargo")
            .render_steps(), @r"
    [build] rustc 0 <host> -> cargo 1 <host>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustdoc 1 <host>
    [build] rustdoc 0 <host>
    [test] rustc 0 <host> -> cargo 1 <host>
    ");
}

#[test]
fn test_cargo_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("test")
            .path("cargo")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> cargo 2 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 2 <host> -> std 2 <host>
    [build] rustdoc 2 <host>
    [build] rustdoc 1 <host>
    [test] rustc 1 <host> -> cargo 2 <host>
    ");
}

#[test]
fn test_cargotest() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("test")
            .path("cargotest")
            .render_steps(), @r"
    [build] rustc 0 <host> -> cargo 1 <host>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 0 <host> -> CargoTest 1 <host>
    [build] rustdoc 1 <host>
    [test] cargotest 1 <host>
    ");
}

#[test]
fn test_tier_check() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("test")
            .path("tier-check")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [test] tier-check <host>
    ");
}

// Differential snapshots for `./x test run-make` run `./x test run-make-cargo`: only
// `run-make-cargo` should build an in-tree cargo, running `./x test run-make` should not.
#[test]
fn test_run_make_no_cargo() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("test")
            .path("run-make")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> RunMakeSupport 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 0 <host> -> Compiletest 1 <host>
    [build] rustdoc 1 <host>
    [test] compiletest-run-make 1 <host>
    ");
}

#[test]
fn test_run_make_cargo_builds_cargo() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("test")
            .path("run-make-cargo")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> RunMakeSupport 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 0 <host> -> Compiletest 1 <host>
    [build] rustc 0 <host> -> cargo 1 <host>
    [build] rustdoc 1 <host>
    [test] compiletest-run-make-cargo 1 <host>
    ");
}

#[test]
fn doc_all() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .render_steps(), @r"
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <host>
    [doc] book (book) <host>
    [doc] book/first-edition (book) <host>
    [doc] book/second-edition (book) <host>
    [doc] book/2018-edition (book) <host>
    [build] rustdoc 0 <host>
    [doc] rustc 0 <host> -> standalone 1 <host>
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [build] rustc 0 <host> -> error-index 1 <host>
    [doc] rustc 0 <host> -> error-index 1 <host>
    [doc] nomicon (book) <host>
    [build] rustc 1 <host> -> std 1 <host>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    [doc] rustdoc (book) <host>
    [doc] rust-by-example (book) <host>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <host>
    [doc] cargo (book) <host>
    [doc] clippy (book) <host>
    [doc] embedded-book (book) <host>
    [doc] edition-guide (book) <host>
    [doc] style-guide (book) <host>
    [doc] rustc 0 <host> -> releases 1 <host>
    ");
}

#[test]
fn doc_library() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .path("library")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    ");
}

#[test]
fn doc_cargo_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .path("cargo")
            .render_steps(), @r"
    [build] rustdoc 0 <host>
    [doc] rustc 0 <host> -> Cargo 1 <host>
    ");
}
#[test]
fn doc_cargo_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .path("cargo")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> Cargo 2 <host>
    ");
}

#[test]
fn doc_core() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .path("core")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[core]
    ");
}

#[test]
fn doc_core_no_std_target() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .path("core")
            .override_target_no_std(&host_target())
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[core]
    ");
}

#[test]
fn doc_library_no_std_target() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .path("library")
            .override_target_no_std(&host_target())
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,core]
    ");
}

#[test]
fn doc_library_no_std_target_cross_compile() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .path("library")
            .targets(&[TEST_TRIPLE_1])
            .override_target_no_std(TEST_TRIPLE_1)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> std 1 <target1> crates=[alloc,core]
    ");
}

#[test]
#[should_panic]
fn doc_compiler_stage_0() {
    let ctx = TestCtx::new();
    ctx.config("doc").path("compiler").stage(0).run();
}

#[test]
fn doc_compiler_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .path("compiler")
            .stage(1)
            .render_steps(), @r"
    [build] rustdoc 0 <host>
    [build] llvm <host>
    [doc] rustc 0 <host> -> rustc 1 <host>
    ");
}

#[test]
fn doc_compiler_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .path("compiler")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> rustc 2 <host>
    ");
}

#[test]
#[should_panic]
fn doc_compiletest_stage_0() {
    let ctx = TestCtx::new();
    ctx.config("doc").path("src/tools/compiletest").stage(0).run();
}

#[test]
fn doc_compiletest_stage_1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .path("src/tools/compiletest")
            .stage(1)
            .render_steps(), @r"
    [build] rustdoc 0 <host>
    [doc] rustc 0 <host> -> Compiletest 1 <host>
    ");
}

#[test]
fn doc_compiletest_stage_2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .path("src/tools/compiletest")
            .stage(2)
            .render_steps(), @r"
    [build] rustdoc 0 <host>
    [doc] rustc 0 <host> -> Compiletest 1 <host>
    ");
}

// Reference should be auto-bumped to stage 2.
#[test]
fn doc_reference() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("doc")
            .path("reference")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    ");
}

#[test]
fn clippy_ci() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("clippy")
            .path("ci")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 0 <host> -> clippy-driver 1 <host>
    [build] rustc 0 <host> -> cargo-clippy 1 <host>
    [clippy] rustc 1 <host> -> bootstrap 2 <host>
    [clippy] rustc 1 <host> -> std 1 <host>
    [clippy] rustc 1 <host> -> rustc 2 <host>
    [check] rustc 1 <host> -> rustc 2 <host>
    [clippy] rustc 1 <host> -> rustc_codegen_gcc 2 <host>
    ");
}

#[test]
fn clippy_compiler_stage1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("clippy")
            .path("compiler")
            .render_steps(), @r"
    [build] llvm <host>
    [clippy] rustc 0 <host> -> rustc 1 <host>
    ");
}

#[test]
fn clippy_compiler_stage2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("clippy")
            .path("compiler")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 0 <host> -> clippy-driver 1 <host>
    [build] rustc 0 <host> -> cargo-clippy 1 <host>
    [clippy] rustc 1 <host> -> rustc 2 <host>
    ");
}

#[test]
fn clippy_std_stage1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("clippy")
            .path("std")
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> clippy-driver 1 <host>
    [build] rustc 0 <host> -> cargo-clippy 1 <host>
    [clippy] rustc 1 <host> -> std 1 <host>
    ");
}

#[test]
fn clippy_std_stage2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("clippy")
            .path("std")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> clippy-driver 2 <host>
    [build] rustc 1 <host> -> cargo-clippy 2 <host>
    [clippy] rustc 2 <host> -> std 2 <host>
    ");
}

#[test]
fn clippy_miri_stage1() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("clippy")
            .path("miri")
            .stage(1)
            .render_steps(), @r"
    [build] llvm <host>
    [check] rustc 0 <host> -> rustc 1 <host>
    [clippy] rustc 0 <host> -> miri 1 <host>
    ");
}

#[test]
fn clippy_miri_stage2() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("clippy")
            .path("miri")
            .stage(2)
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 1 <host> -> std 1 <host>
    [check] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 0 <host> -> clippy-driver 1 <host>
    [build] rustc 0 <host> -> cargo-clippy 1 <host>
    [clippy] rustc 1 <host> -> miri 2 <host>
    ");
}

#[test]
fn clippy_bootstrap() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("clippy")
            .path("bootstrap")
            .render_steps(), @"[clippy] rustc 0 <host> -> bootstrap 1 <host>");
}

#[test]
fn install_extended() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("install")
            .args(&[
                // Using backslashes fails with `--set`
                "--set", &format!("install.prefix={}", ctx.dir().display()).replace("\\", "/"),
                "--set", &format!("install.sysconfdir={}", ctx.dir().display()).replace("\\", "/"),
                "--set", "build.extended=true"
            ])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> WasmComponentLd 1 <host>
    [build] rustc 0 <host> -> UnstableBookGen 1 <host>
    [build] rustc 0 <host> -> Rustbook 1 <host>
    [doc] unstable-book (book) <host>
    [build] rustc 1 <host> -> std 1 <host>
    [doc] book (book) <host>
    [doc] book/first-edition (book) <host>
    [doc] book/second-edition (book) <host>
    [doc] book/2018-edition (book) <host>
    [build] rustdoc 1 <host>
    [doc] rustc 1 <host> -> standalone 2 <host>
    [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
    [build] rustc 1 <host> -> rustc 2 <host>
    [build] rustc 1 <host> -> WasmComponentLd 2 <host>
    [build] rustc 1 <host> -> error-index 2 <host>
    [doc] rustc 1 <host> -> error-index 2 <host>
    [doc] nomicon (book) <host>
    [doc] rustc 1 <host> -> reference (book) 2 <host>
    [doc] rustdoc (book) <host>
    [doc] rust-by-example (book) <host>
    [build] rustc 0 <host> -> LintDocs 1 <host>
    [doc] rustc (book) <host>
    [doc] cargo (book) <host>
    [doc] clippy (book) <host>
    [doc] embedded-book (book) <host>
    [doc] edition-guide (book) <host>
    [doc] style-guide (book) <host>
    [doc] rustc 1 <host> -> releases 2 <host>
    [build] rustc 0 <host> -> RustInstaller 1 <host>
    [dist] docs <host>
    [dist] rustc 1 <host> -> std 1 <host>
    [build] rustdoc 2 <host>
    [build] rustc 1 <host> -> rust-analyzer-proc-macro-srv 2 <host>
    [build] rustc 0 <host> -> GenerateCopyright 1 <host>
    [dist] rustc <host>
    [build] rustc 1 <host> -> cargo 2 <host>
    [dist] rustc 1 <host> -> cargo 2 <host>
    [build] rustc 1 <host> -> rust-analyzer 2 <host>
    [dist] rustc 1 <host> -> rust-analyzer 2 <host>
    [build] rustc 1 <host> -> rustfmt 2 <host>
    [build] rustc 1 <host> -> cargo-fmt 2 <host>
    [dist] rustc 1 <host> -> rustfmt 2 <host>
    [build] rustc 1 <host> -> clippy-driver 2 <host>
    [build] rustc 1 <host> -> cargo-clippy 2 <host>
    [dist] rustc 1 <host> -> clippy 2 <host>
    [build] rustc 1 <host> -> miri 2 <host>
    [build] rustc 1 <host> -> cargo-miri 2 <host>
    [dist] rustc 1 <host> -> miri 2 <host>
    [dist] src <>
    ");
}

// Check that `x run miri --target FOO` actually builds miri for the host.
#[test]
fn run_miri() {
    let ctx = TestCtx::new();
    insta::assert_snapshot!(
        ctx.config("run")
            .path("miri")
            .stage(1)
            .targets(&[TEST_TRIPLE_1])
            .render_steps(), @r"
    [build] llvm <host>
    [build] rustc 0 <host> -> rustc 1 <host>
    [build] rustc 0 <host> -> miri 1 <host>
    [build] rustc 0 <host> -> cargo-miri 1 <host>
    [run] rustc 0 <host> -> miri 1 <target1>
    ");
}
