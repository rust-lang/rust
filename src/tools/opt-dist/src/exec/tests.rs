use super::*;

#[test]
fn llvm_profile_cache_input() {
    let build = || Bootstrap { cmd: CmdBuilder::default(), metrics_path: Utf8PathBuf::new() };
    let inherited = std::env::var_os("SCCACHE_EXTRAFILES");
    let profile = LlvmPGOProfile(Utf8PathBuf::from("profile dir;cache/llvm.profdata"));
    let optimized = build().llvm_pgo_optimize(Some(&profile)).unwrap();
    let mut paths =
        std::env::split_paths(&optimized.cmd.env["SCCACHE_EXTRAFILES"]).collect::<Vec<_>>();
    assert_eq!(paths.pop(), Some(profile.0.into_std_path_buf()));
    assert_eq!(
        paths,
        inherited.as_ref().map(std::env::split_paths).into_iter().flatten().collect::<Vec<_>>()
    );
    assert!(build().llvm_pgo_optimize(None).unwrap().cmd.env.is_empty());

    let invalid_profile = LlvmPGOProfile(Utf8PathBuf::from("invalid\"profile"));
    let error = build().llvm_pgo_optimize(Some(&invalid_profile)).err().unwrap();
    assert_eq!(error.to_string(), "SCCACHE_EXTRAFILES");
    assert!(error.downcast_ref::<std::env::JoinPathsError>().is_some());
}
