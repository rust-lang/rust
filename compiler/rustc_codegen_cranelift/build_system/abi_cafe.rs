use crate::path::{Dirs, RelPath};
use crate::prepare::GitRepo;
use crate::utils::{CargoProject, Compiler, spawn_and_wait};
use crate::{CodegenBackend, SysrootKind, build_sysroot};

static ABI_CAFE_REPO: GitRepo = GitRepo::github(
    "Gankra",
    "abi-cafe",
    "f1220cfd13b57f5c0082c26529163865ee25e115",
    "fe93a9acd461425d",
    "abi-cafe",
);

static ABI_CAFE: CargoProject = CargoProject::new(&ABI_CAFE_REPO.source_dir(), "abi_cafe_target");

pub(crate) fn run(
    sysroot_kind: SysrootKind,
    dirs: &Dirs,
    cg_clif_dylib: &CodegenBackend,
    rustup_toolchain_name: Option<&str>,
    bootstrap_host_compiler: &Compiler,
) {
    RelPath::DOWNLOAD.ensure_exists(dirs);
    ABI_CAFE_REPO.fetch(dirs);
    ABI_CAFE_REPO.patch(dirs);

    eprintln!("Building sysroot for abi-cafe");
    build_sysroot::build_sysroot(
        dirs,
        sysroot_kind,
        cg_clif_dylib,
        bootstrap_host_compiler,
        rustup_toolchain_name,
        bootstrap_host_compiler.triple.clone(),
    );

    eprintln!("Running abi-cafe");

    let pairs: &[_] =
        if cfg!(not(any(target_os = "macos", all(target_os = "windows", target_env = "msvc")))) {
            &["rustc_calls_cgclif", "cgclif_calls_rustc", "cgclif_calls_cc", "cc_calls_cgclif"]
        } else {
            &["rustc_calls_cgclif", "cgclif_calls_rustc"]
        };

    let mut cmd = ABI_CAFE.run(bootstrap_host_compiler, dirs);
    cmd.arg("--");

    // stdcall, vectorcall and such don't work yet
    cmd.arg("--conventions").arg("c").arg("--conventions").arg("rust");

    for pair in pairs {
        cmd.arg("--pairs").arg(pair);
    }

    cmd.arg("--add-rustc-codegen-backend");
    match cg_clif_dylib {
        CodegenBackend::Local(path) => {
            cmd.arg(format!("cgclif:{}", path.display()));
        }
        CodegenBackend::Builtin(name) => {
            cmd.arg(format!("cgclif:{name}"));
        }
    }

    cmd.current_dir(ABI_CAFE.source_dir(dirs));

    spawn_and_wait(cmd);
}
