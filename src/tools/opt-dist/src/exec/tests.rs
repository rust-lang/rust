use super::*;
use crate::environment::EnvironmentBuilder;

#[test]
fn llvm_training_rebuilds_only_static_rustc() {
    for (host, shared) in [
        ("x86_64-unknown-linux-gnu", true),
        ("aarch64-unknown-linux-gnu", true),
        ("x86_64-pc-windows-msvc", false),
        ("aarch64-pc-windows-msvc", false),
    ] {
        for fast_try_build in [false, true] {
            let env = EnvironmentBuilder::default()
                .host_tuple(host.to_owned())
                .python_binary("python".to_owned())
                .checkout_dir(Utf8PathBuf::from("checkout"))
                .build_dir(Utf8PathBuf::from("build"))
                .artifact_dir(Utf8PathBuf::from("profiles"))
                .host_llvm_dir(Utf8PathBuf::from("clang"))
                .skipped_tests(vec![])
                .use_bolt(false)
                .shared_llvm(shared)
                .run_tests(true)
                .fast_try_build(fast_try_build)
                .build_llvm(true)
                .build()
                .unwrap();
            let profile = RustcPGOProfile(Utf8PathBuf::from("profiles/rustc.profdata"));
            let bootstrap =
                Bootstrap::llvm_pgo_instrument(&env, Utf8Path::new("profiles/llvm"), &profile);
            let mut expected = Bootstrap::build(&env).cmd.args;
            expected
                .extend(["--set".into(), r#"pgo.llvm.generate="profiles/llvm/prof-%p""#.into()]);
            if !shared {
                expected
                    .extend(["--set".into(), r#"pgo.rustc.use="profiles/rustc.profdata""#.into()]);
            }
            expected.extend(["--keep-stage".into(), "0".into()]);
            if shared {
                expected.extend(["--keep-stage".into(), "1".into()]);
            }
            assert_eq!(bootstrap.cmd.args, expected, "{host}, fast_try_build={fast_try_build}");
        }
    }
}
