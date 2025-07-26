use std::path::{Path, PathBuf};
use std::{env, iter};

use super::*;
use crate::core::config::{Target, TargetSelection};
use crate::{Build, Config, Flags};

#[test]
fn test_ndk_compiler_c() {
    let ndk_path = PathBuf::from("/ndk");
    let target_triple = "arm-unknown-linux-android";
    let expected_triple_translated = "armv7a-unknown-linux-android";
    let expected_compiler = format!("{}21-{}", expected_triple_translated, Language::C.clang());
    let host_tag = if cfg!(target_os = "macos") {
        "darwin-x86_64"
    } else if cfg!(target_os = "windows") {
        "windows-x86_64"
    } else {
        "linux-x86_64"
    };
    let expected_path = ndk_path
        .join("toolchains")
        .join("llvm")
        .join("prebuilt")
        .join(host_tag)
        .join("bin")
        .join(&expected_compiler);
    let result = ndk_compiler(Language::C, target_triple, &ndk_path);
    assert_eq!(result, expected_path);
}

#[test]
fn test_ndk_compiler_cpp() {
    let ndk_path = PathBuf::from("/ndk");
    let target_triple = "arm-unknown-linux-android";
    let expected_triple_translated = "armv7a-unknown-linux-android";
    let expected_compiler =
        format!("{}21-{}", expected_triple_translated, Language::CPlusPlus.clang());
    let host_tag = if cfg!(target_os = "macos") {
        "darwin-x86_64"
    } else if cfg!(target_os = "windows") {
        "windows-x86_64"
    } else {
        "linux-x86_64"
    };
    let expected_path = ndk_path
        .join("toolchains")
        .join("llvm")
        .join("prebuilt")
        .join(host_tag)
        .join("bin")
        .join(&expected_compiler);
    let result = ndk_compiler(Language::CPlusPlus, target_triple, &ndk_path);
    assert_eq!(result, expected_path);
}

#[test]
fn test_language_gcc() {
    assert_eq!(Language::C.gcc(), "gcc");
    assert_eq!(Language::CPlusPlus.gcc(), "g++");
}

#[test]
fn test_language_clang() {
    assert_eq!(Language::C.clang(), "clang");
    assert_eq!(Language::CPlusPlus.clang(), "clang++");
}

#[test]
fn test_new_cc_build() {
    let build = Build::new(Config { ..Config::parse(Flags::parse(&["build".to_owned()])) });
    let target = TargetSelection::from_user("x86_64-unknown-linux-gnu");
    let cfg = new_cc_build(&build, target.clone());
    let compiler = cfg.get_compiler();
    assert!(!compiler.path().to_str().unwrap().is_empty(), "Compiler path should not be empty");
}

#[test]
fn test_default_compiler_wasi() {
    let mut build = Build::new(Config { ..Config::parse(Flags::parse(&["build".to_owned()])) });
    let target = TargetSelection::from_user("wasm32-wasi");
    let wasi_sdk = PathBuf::from("/wasi-sdk");
    build.wasi_sdk_path = Some(wasi_sdk.clone());

    let mut cfg = cc::Build::new();
    if let Some(result) = default_compiler(&mut cfg, Language::C, target.clone(), &build) {
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
    let build = Build::new(Config { ..Config::parse(Flags::parse(&["build".to_owned()])) });
    let target = TargetSelection::from_user("x86_64-unknown-linux-gnu");
    let mut cfg = cc::Build::new();
    let result = default_compiler(&mut cfg, Language::C, target, &build);
    assert!(result.is_none(), "default_compiler should return None for generic targets");
}

#[test]
fn test_find_target_with_config() {
    let mut build = Build::new(Config { ..Config::parse(Flags::parse(&["build".to_owned()])) });
    let target = TargetSelection::from_user("x86_64-unknown-linux-gnu");
    let mut target_config = Target::default();
    target_config.cc = Some(PathBuf::from("dummy-cc"));
    target_config.cxx = Some(PathBuf::from("dummy-cxx"));
    target_config.ar = Some(PathBuf::from("dummy-ar"));
    target_config.ranlib = Some(PathBuf::from("dummy-ranlib"));
    build.config.target_config.insert(target.clone(), target_config);
    fill_target_compiler(&mut build, target.clone());
    let cc_tool = build.cc.get(&target).unwrap();
    assert_eq!(cc_tool.path(), &PathBuf::from("dummy-cc"));
    let cxx_tool = build.cxx.get(&target).unwrap();
    assert_eq!(cxx_tool.path(), &PathBuf::from("dummy-cxx"));
    let ar = build.ar.get(&target).unwrap();
    assert_eq!(ar, &PathBuf::from("dummy-ar"));
    let ranlib = build.ranlib.get(&target).unwrap();
    assert_eq!(ranlib, &PathBuf::from("dummy-ranlib"));
}

#[test]
fn test_find_target_without_config() {
    let mut build = Build::new(Config { ..Config::parse(Flags::parse(&["build".to_owned()])) });
    let target = TargetSelection::from_user("x86_64-unknown-linux-gnu");
    build.config.target_config.clear();
    fill_target_compiler(&mut build, target.clone());
    assert!(build.cc.contains_key(&target));
    if !target.triple.contains("vxworks") {
        assert!(build.cxx.contains_key(&target));
    }
    assert!(build.ar.contains_key(&target));
}

#[test]
fn test_find() {
    let mut build = Build::new(Config { ..Config::parse(Flags::parse(&["build".to_owned()])) });
    let target1 = TargetSelection::from_user("x86_64-unknown-linux-gnu");
    let target2 = TargetSelection::from_user("x86_64-unknown-openbsd");
    build.targets.push(target1.clone());
    build.hosts.push(target2.clone());
    fill_compilers(&mut build);
    for t in build.hosts.iter().chain(build.targets.iter()).chain(iter::once(&build.host_target)) {
        assert!(build.cc.contains_key(t), "CC not set for target {}", t.triple);
    }
}
