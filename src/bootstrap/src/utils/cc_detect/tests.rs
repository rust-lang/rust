use std::iter;
use std::path::PathBuf;

use super::*;
use crate::core::config::{Target, TargetSelection};
use crate::core::session::Session;
use crate::utils::tests::TestCtx;

#[test]
fn test_ndk_compiler() {
    let ndk_path = PathBuf::from("/ndk");
    let host_tag = if cfg!(target_os = "macos") {
        "darwin-x86_64"
    } else if cfg!(target_os = "windows") {
        "windows-x86_64"
    } else {
        "linux-x86_64"
    };

    for (language, compiler) in [
        (Language::C, "armv7a-unknown-linux-android21-clang"),
        (Language::CPlusPlus, "armv7a-unknown-linux-android21-clang++"),
    ] {
        let expected =
            ndk_path.join("toolchains/llvm/prebuilt").join(host_tag).join("bin").join(compiler);

        assert_eq!(ndk_compiler(language, "arm-unknown-linux-android", &ndk_path), expected);
    }
}

#[test]
fn test_language_compiler_names() {
    assert_eq!(Language::C.gcc(), "gcc");
    assert_eq!(Language::CPlusPlus.gcc(), "g++");

    assert_eq!(Language::C.clang(), "clang");
    assert_eq!(Language::CPlusPlus.clang(), "clang++");
}

#[test]
fn test_new_cc_build() {
    let config = TestCtx::new().config("build").create_config();
    let sess = Session::new(config);
    let target = TargetSelection::from_user("x86_64-unknown-linux-gnu");
    let cfg = new_cc_build(&sess, target.clone());
    let compiler = cfg.get_compiler();
    assert!(!compiler.path().to_str().unwrap().is_empty(), "Compiler path should not be empty");
}

#[test]
fn test_default_compiler_wasi() {
    let config = TestCtx::new().config("build").create_config();
    let mut sess = Session::new(config);
    let target = TargetSelection::from_user("wasm32-wasi");
    let wasi_sdk = PathBuf::from("/wasi-sdk");
    sess.wasi_sdk_path = Some(wasi_sdk.clone());

    let cfg = cc::Build::new();
    if let Some(result) = default_compiler(&cfg, Language::C, target.clone(), &sess) {
        let expected = {
            let compiler = format!("{}-clang", target.triple);
            wasi_sdk.join("bin").join(compiler)
        };
        assert_eq!(result, expected);
    } else {
        panic!(
            "default_compiler should return a compiler path for wasi target when WASI_SDK_PATH is set"
        );
    }
}

#[test]
fn test_default_compiler_fallback() {
    let config = TestCtx::new().config("build").create_config();
    let sess = Session::new(config);
    let target = TargetSelection::from_user("x86_64-unknown-linux-gnu");
    let cfg = cc::Build::new();
    let result = default_compiler(&cfg, Language::C, target, &sess);
    assert!(result.is_none(), "default_compiler should return None for generic targets");
}

#[test]
fn test_find_target_with_config() {
    let config = TestCtx::new().config("build").create_config();
    let mut sess = Session::new(config);
    let target = TargetSelection::from_user("x86_64-unknown-linux-gnu");
    let mut target_config = Target::default();
    target_config.cc = Some(PathBuf::from("dummy-cc"));
    target_config.cxx = Some(PathBuf::from("dummy-cxx"));
    target_config.ar = Some(PathBuf::from("dummy-ar"));
    target_config.ranlib = Some(PathBuf::from("dummy-ranlib"));
    sess.config.target_config.insert(target.clone(), target_config);
    fill_target_compiler(&mut sess, target.clone());
    let cc_tool = sess.cc.get(&target).unwrap();
    assert_eq!(cc_tool.path(), &PathBuf::from("dummy-cc"));
    let cxx_tool = sess.cxx.get(&target).unwrap();
    assert_eq!(cxx_tool.path(), &PathBuf::from("dummy-cxx"));
    let ar = sess.ar.get(&target).unwrap();
    assert_eq!(ar, &PathBuf::from("dummy-ar"));
    let ranlib = sess.ranlib.get(&target).unwrap();
    assert_eq!(ranlib, &PathBuf::from("dummy-ranlib"));
}

#[test]
fn test_find_target_without_config() {
    let config = TestCtx::new().config("build").create_config();
    let mut sess = Session::new(config);
    let target = TargetSelection::from_user("x86_64-unknown-linux-gnu");
    sess.config.target_config.clear();
    fill_target_compiler(&mut sess, target.clone());
    assert!(sess.cc.contains_key(&target));
    if !target.triple.contains("vxworks") {
        assert!(sess.cxx.contains_key(&target));
    }
    assert!(sess.ar.contains_key(&target));
}

#[test]
fn test_find() {
    let config = TestCtx::new().config("build").create_config();
    let mut sess = Session::new(config);
    let target1 = TargetSelection::from_user("x86_64-unknown-linux-gnu");
    let target2 = TargetSelection::from_user("x86_64-unknown-openbsd");
    sess.config.targets.push(target1.clone());
    sess.config.hosts.push(target2.clone());
    fill_compilers(&mut sess);
    for t in sess.hosts.iter().chain(sess.targets.iter()).chain(iter::once(&sess.host_target)) {
        assert!(sess.cc.contains_key(t), "CC not set for target {}", t.triple);
    }
}
