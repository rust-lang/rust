use std::path::PathBuf;

use crate::core::build_steps::compile;
use crate::core::build_steps::compile::{apple_darwin_sign_file, apple_darwin_update_library_name};
use crate::core::build_steps::llvm::{
    LdFlags, Llvm, LlvmResult, SanitizerRuntime, common_sanitizer_lib, configure_cmake,
    darwin_sanitizer_lib,
};
use crate::core::build_steps::tool::{SourceType, prepare_tool_cargo};
use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::helpers::is_dylib;
use crate::utils::exec::command;
use crate::*;

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
        run.path("src/tools/bsan/bsan-rt").default_condition(
            builder.config.extended
                && builder.config.tools.as_ref().map_or(
                    builder.build.unstable_features(),
                    |tools| {
                        tools.iter().any(|tool: &String| match tool.as_str() {
                            x => "bsan-rt" == x,
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

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let target = self.target;
        let compiler = self.compiler;

        if !builder.config.llvm_tools_enabled {
            eprintln!("ERROR: LLVM tools must be enabled to build BorrowSanitizer.");
            exit!(1);
        }

        let bindir = builder.sysroot_target_bindir(compiler, target);
        let out_dir = builder.native_dir(self.target).join("sanitizers");

        let cpp_runtime = supports_bsan(&out_dir, self.target, &builder.config.channel);
        if cpp_runtime.is_none() {
            return None;
        }
        let cpp_runtime = cpp_runtime.unwrap();
        let LlvmResult { llvm_config, .. } = builder.ensure(Llvm { target: builder.config.build });

        let compiler_rt_dir = builder.src.join("src/llvm-project/compiler-rt");
        if !compiler_rt_dir.exists() {
            return None;
        }

        let rust_runtime_path = builder.ensure(BsanRTCore { compiler, target });
        let rust_runtime_parent_dir = rust_runtime_path.parent().unwrap();

        if builder.config.dry_run() {
            return Some(cpp_runtime);
        }
        // On targets that build BSAN as a static runtime, we need to manually add in the object files
        // for the Rust runtime using llvm-ar (see below). If the C++ sources haven't changed, then CMake
        // will not rebuild or relink the C++ component. However,  the name and quanity of the objects from
        // the Rust runtimeis subject to change, so there's no way to detect if a prebuilt copy of the C++
        // runtime already contains objects from a prior build of the Rust component. To avoid name clashes
        // due to this issue, we just ensure that the C++ runtime is relinked fresh each time, which
        // is relatively inexpensive.
        if cpp_runtime.path.exists() {
            fs::remove_file(&cpp_runtime.path).unwrap();
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

        let mut ldflags = LdFlags::default();
        ldflags.push_all(format!("-L{}", &rust_runtime_parent_dir.display()));

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

        cfg.build_target(&cpp_runtime.cmake_target);
        cfg.build();

        // If we're building BSAN as a static library, then we need to manually
        // patch-in our Rust component, since there doesn't appear to be a straightforward
        // way to declare an external static archive (libbsan_rt.a) as a build dependency
        // of another static archive (libclang-rt-<arch>.bsan.a) in CMake—at least, not if
        // the external archive contains an unknown, varying quantity of object files.
        if !is_dylib(&cpp_runtime.name) {
            let temp_dir = builder.build.tempdir().join("bsan-rt");
            if temp_dir.exists() {
                fs::remove_dir_all(&temp_dir).unwrap();
            }
            fs::create_dir_all(&temp_dir).unwrap();

            // Since our Rust runtime depends on core,
            // we need to remove all global symbols except for
            // our API endpoints to avoid clashing with users' programs.
            command(bindir.join(exe("llvm-objcopy", compiler.host)))
                .current_dir(&temp_dir)
                .arg("-w")
                .arg("--keep-global-symbol=bsan_*")
                .arg(&rust_runtime_path)
                .run(builder);

            // Then, we unpack the Rust archive into a collection of object files.
            // It *would* be possible to list these files as a CMake dependency, but that
            // would break rebuilds, since the name and quantity of object files in the archive
            // will change throughout development.
            let llvm_ar = bindir.join(exe("llvm-ar", compiler.host));
            command(&llvm_ar).current_dir(&temp_dir).arg("-x").arg(rust_runtime_path).run(builder);
            let file_names: Vec<String> = fs::read_dir(&temp_dir)
                .unwrap()
                .filter_map(|entry| {
                    let path = entry.ok().unwrap().path();
                    if path.is_file() { path.to_str().map(|s| s.to_owned()) } else { None }
                })
                .collect();

            // Finally, add the objects into the static archive of C++ component.
            command(llvm_ar)
                .current_dir(&temp_dir)
                .arg("-r")
                .arg(&cpp_runtime.path)
                .args(file_names)
                .run(builder);
        }

        let libdir = builder.sysroot_target_libdir(compiler, target);
        let dst = libdir.join(&cpp_runtime.name);
        builder.copy_link(&cpp_runtime.path, &dst);

        if target.contains("-apple-") {
            // Update the library’s install name to reflect that it has been renamed.
            apple_darwin_update_library_name(
                builder,
                &dst,
                &format!("@rpath/{}", &cpp_runtime.name),
            );
            // Upon renaming the install name, the code signature of the file will invalidate,
            // so we will sign it again.
            apple_darwin_sign_file(builder, &dst);
        }
        Some(cpp_runtime)
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
        run.never()
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
            "src/tools/bsan/bsan-rt",
            SourceType::InTree,
            &Vec::new(),
        );

        cargo.rustflag("-Cembed-bitcode=yes");
        cargo.rustflag("-Clto");
        cargo.rustflag("-Cpanic=abort");
        cargo.env("BSAN_HEADER_DIR", builder.cargo_out(compiler, mode, target));
        let build_success = compile::stream_cargo(builder, cargo, vec![], &mut |_| {});
        if !build_success {
            crate::exit!(1);
        } else {
            builder.cargo_out(compiler, mode, target).join("libbsan_rt.a")
        }
    }
}
