use std::path::PathBuf;

use super::llvm;
use crate::core::build_steps::compile;
use crate::core::build_steps::compile::{apple_darwin_sign_file, apple_darwin_update_library_name};
use crate::core::build_steps::llvm::{
    LdFlags, Llvm, LlvmResult, SanitizerRuntime, common_sanitizer_lib, configure_cmake,
    darwin_sanitizer_lib,
};
use crate::core::build_steps::tool::{SourceType, prepare_tool_cargo};
use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::helpers::dylib;
use crate::{Compiler, Kind, Mode, Path, fs, helpers, t};


const BSAN_CORE_PATH: &str = "src/tools/bsan/bsanrt";
const BSAN_RT_DYLIB: &str = "bsanrt";


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BsanRT {
    pub compiler: Compiler,
    pub target: TargetSelection,
}
impl Step for BsanRT {
    type Output = Option<SanitizerRuntime>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;
    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path(BSAN_CORE_PATH).default_condition(
            builder.config.extended
                && builder.config.tools.as_ref().map_or(
                    builder.build.unstable_features(),
                    |tools| {
                        tools.iter().any(|tool: &String| match tool.as_str() {
                            x => BSAN_RT_DYLIB == x,
                        })
                    },
                ),
        )
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(BsanRT {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
        });
    }

    /// Builds sanitizer runtime libraries.
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let target = self.target;
        let compiler = self.compiler;

        builder.ensure(llvm::Llvm { target });
        builder.ensure(BsanRTCore { compiler, target });

        let compiler_rt_dir = builder.src.join("src/llvm-project/compiler-rt");
        if !compiler_rt_dir.exists() {
            return None;
        }

        let out_dir = builder.native_dir(self.target).join("sanitizers");

        let runtime = supports_bsan(&out_dir, self.target, &builder.config.channel);
        if runtime.is_none() {
            return None;
        }
        let runtime = runtime.unwrap();

        let LlvmResult { llvm_config, .. } = builder.ensure(Llvm { target: builder.config.build });
        if builder.config.dry_run() {
            return Some(runtime);
        }
        let _guard = builder.msg_unstaged(Kind::Build, "bsan", self.target);
        let _time = helpers::timeit(builder);

        let mut cfg = cmake::Config::new(&compiler_rt_dir);
        cfg.profile("Release");
        cfg.define("CMAKE_C_COMPILER_TARGET", self.target.triple);
        cfg.define("COMPILER_RT_BUILD_BUILTINS", "OFF");
        cfg.define("COMPILER_RT_BUILD_CRT", "OFF");
        cfg.define("COMPILER_RT_BUILD_LIBFUZZER", "OFF");
        cfg.define("COMPILER_RT_BUILD_PROFILE", "OFF");
        cfg.define("COMPILER_RT_BUILD_SANITIZERS", "ON");
        cfg.define("COMPILER_RT_BUILD_XRAY", "OFF");
        cfg.define("COMPILER_RT_DEFAULT_TARGET_ONLY", "ON");
        cfg.define("COMPILER_RT_USE_LIBCXX", "OFF");
        cfg.define("LLVM_CONFIG_PATH", &llvm_config);
        // On Darwin targets the sanitizer runtimes are build as universal binaries.
        // Unfortunately sccache currently lacks support to build them successfully.
        // Disable compiler launcher on Darwin targets to avoid potential issues.
        let use_compiler_launcher = !self.target.contains("apple-darwin");
        // Since v1.0.86, the cc crate adds -mmacosx-version-min to the default
        // flags on MacOS. A long-standing bug in the CMake rules for compiler-rt
        // causes architecture detection to be skipped when this flag is present,
        // and compilation fails. https://github.com/llvm/llvm-project/issues/88780
        let suppressed_compiler_flag_prefixes: &[&str] =
            if self.target.contains("apple-darwin") { &["-mmacosx-version-min="] } else { &[] };

        let sysroot = &builder.sysroot_target_libdir(self.compiler, self.target);
        let sysroot = sysroot.display();
        let mut ldflags = LdFlags::default();
        ldflags.push_all(format!("-L{sysroot}"));

        configure_cmake(
            builder,
            self.target,
            &mut cfg,
            use_compiler_launcher,
            ldflags,
            suppressed_compiler_flag_prefixes,
        );

        t!(fs::create_dir_all(&out_dir));
        cfg.out_dir(out_dir);

        cfg.build_target(&runtime.cmake_target);
        cfg.build();

        let libdir = builder.sysroot_target_libdir(compiler, target);
        let dst = libdir.join(&runtime.name);
        builder.copy_link(&runtime.path, &dst);

        if target == "x86_64-apple-darwin"
            || target == "aarch64-apple-darwin"
            || target == "aarch64-apple-ios"
            || target == "aarch64-apple-ios-sim"
            || target == "x86_64-apple-ios"
        {
            // Update the library’s install name to reflect that it has been renamed.
            apple_darwin_update_library_name(builder, &dst, &format!("@rpath/{}", runtime.name));
            // Upon renaming the install name, the code signature of the file will invalidate,
            // so we will sign it again.
            apple_darwin_sign_file(builder, &dst);
        }

        Some(runtime)
    }
}

pub fn supports_bsan(
    out_dir: &Path,
    target: TargetSelection,
    channel: &str,
) -> Option<SanitizerRuntime> {
    match &*target.triple {
        "aarch64-apple-darwin" => Some(darwin_sanitizer_lib("bsan", "osx", channel, out_dir)),
        "aarch64-unknown-linux-gnu" => {
            Some(common_sanitizer_lib("bsan", "linux", "aarch64", channel, out_dir))
        }
        "x86_64-unknown-linux-gnu" => {
            Some(common_sanitizer_lib("bsan", "linux", "x86_64", channel, out_dir))
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BsanRTCore {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for BsanRTCore {
    type Output = PathBuf;
    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/bsan/bsanrt")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(BsanRTCore {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let compiler = self.compiler;
        let target = self.target;
        let mode = Mode::ToolRustc;
        let kind = Kind::Build;

        builder.ensure(compile::Rustc::new(compiler, target));

        let mut cargo = prepare_tool_cargo(
            builder,
            compiler,
            mode,
            target,
            kind,
            BSAN_CORE_PATH,
            SourceType::InTree,
            &Vec::new(),
        );
        cargo.env("BSAN_HEADER_DIR", builder.cargo_out(compiler, mode, target));

        // we check this below
        let build_success = compile::stream_cargo(builder, cargo, vec![], &mut |_| {});
        if !build_success {
            crate::exit!(1);
        } else {
            let file_name = dylib(BSAN_RT_DYLIB, target);
            let runtime = builder.cargo_out(compiler, mode, target).join(&file_name);
            if target == "x86_64-apple-darwin"
                || target == "aarch64-apple-darwin"
                || target == "aarch64-apple-ios"
                || target == "aarch64-apple-ios-sim"
                || target == "x86_64-apple-ios"
            {
                // Update the library’s install name to reflect that it has been renamed.
                apple_darwin_update_library_name(
                    builder,
                    &runtime,
                    &format!("@rpath/{}", file_name),
                );
                // Upon renaming the install name, the code signature of the file will invalidate,
                // so we will sign it again.
                apple_darwin_sign_file(builder, &runtime);
            }
            let libdir = builder.sysroot_target_libdir(compiler, target);
            let dst = libdir.join(dylib(BSAN_RT_DYLIB, target));
            builder.copy_link(&runtime, &dst);
            dst
        }
    }
}

