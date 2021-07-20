use std::path::PathBuf;

use base_db::FileId;
use expect_test::{expect, Expect};

use crate::{CargoWorkspace, CfgOverrides, ProjectWorkspace, Sysroot, WorkspaceBuildScripts};

fn check(file: &str, expect: Expect) {
    let meta = get_test_metadata(file);
    let cargo_workspace = CargoWorkspace::new(meta);
    let project_workspace = ProjectWorkspace::Cargo {
        cargo: cargo_workspace,
        build_scripts: WorkspaceBuildScripts::default(),
        sysroot: Sysroot::default(),
        rustc: None,
        rustc_cfg: Vec::new(),
        cfg_overrides: CfgOverrides::default(),
    };

    let crate_graph = project_workspace.to_crate_graph(None, {
        let mut counter = 0;
        &mut move |_path| {
            counter += 1;
            Some(FileId(counter))
        }
    });

    let mut crate_graph = format!("{:#?}", crate_graph);
    replace_root(&mut crate_graph, false);

    expect.assert_eq(&crate_graph);
}

fn get_test_metadata(file: &str) -> cargo_metadata::Metadata {
    let mut json = get_test_data(file).parse::<serde_json::Value>().unwrap();
    fixup_paths(&mut json);
    return serde_json::from_value(json).unwrap();

    fn fixup_paths(val: &mut serde_json::Value) -> () {
        match val {
            serde_json::Value::String(s) => replace_root(s, true),
            serde_json::Value::Array(vals) => vals.iter_mut().for_each(fixup_paths),
            serde_json::Value::Object(kvals) => kvals.values_mut().for_each(fixup_paths),
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
                ()
            }
        }
    }
}

fn replace_root(s: &mut String, direction: bool) {
    if direction {
        let root = if cfg!(windows) { r#"C:\\ROOT\"# } else { "/ROOT/" };
        *s = s.replace("$ROOT$", root)
    } else {
        let root = if cfg!(windows) { r#"C:\\\\ROOT\\"# } else { "/ROOT/" };
        *s = s.replace(root, "$ROOT$")
    }
}

fn get_test_data(file: &str) -> String {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let file = base.join("test_data").join(file);
    std::fs::read_to_string(file).unwrap()
}

#[test]
fn hello_world_project_model() {
    check(
        "hello-world-metadata.json",
        expect![[r#"
            CrateGraph {
                arena: {
                    CrateId(
                        0,
                    ): CrateData {
                        root_file_id: FileId(
                            1,
                        ),
                        edition: Edition2018,
                        display_name: Some(
                            CrateDisplayName {
                                crate_name: CrateName(
                                    "hello_world",
                                ),
                                canonical_name: "hello-world",
                            },
                        ),
                        cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "test",
                            ],
                        ),
                        potential_cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "test",
                            ],
                        ),
                        env: Env {
                            entries: {
                                "CARGO_PKG_LICENSE": "",
                                "CARGO_PKG_VERSION_MAJOR": "0",
                                "CARGO_MANIFEST_DIR": "$ROOT$hello-world",
                                "CARGO_PKG_VERSION": "0.1.0",
                                "CARGO_PKG_AUTHORS": "",
                                "CARGO_CRATE_NAME": "hello_world",
                                "CARGO_PKG_LICENSE_FILE": "",
                                "CARGO_PKG_HOMEPAGE": "",
                                "CARGO_PKG_DESCRIPTION": "",
                                "CARGO_PKG_NAME": "hello-world",
                                "CARGO_PKG_VERSION_PATCH": "0",
                                "CARGO": "cargo",
                                "CARGO_PKG_REPOSITORY": "",
                                "CARGO_PKG_VERSION_MINOR": "1",
                                "CARGO_PKG_VERSION_PRE": "",
                            },
                        },
                        dependencies: [
                            Dependency {
                                crate_id: CrateId(
                                    4,
                                ),
                                name: CrateName(
                                    "libc",
                                ),
                            },
                        ],
                        proc_macro: [],
                    },
                    CrateId(
                        5,
                    ): CrateData {
                        root_file_id: FileId(
                            6,
                        ),
                        edition: Edition2015,
                        display_name: Some(
                            CrateDisplayName {
                                crate_name: CrateName(
                                    "const_fn",
                                ),
                                canonical_name: "const_fn",
                            },
                        ),
                        cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "feature=default",
                                "feature=std",
                                "test",
                            ],
                        ),
                        potential_cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "feature=align",
                                "feature=const-extern-fn",
                                "feature=default",
                                "feature=extra_traits",
                                "feature=rustc-dep-of-std",
                                "feature=std",
                                "feature=use_std",
                                "test",
                            ],
                        ),
                        env: Env {
                            entries: {
                                "CARGO_PKG_LICENSE": "",
                                "CARGO_PKG_VERSION_MAJOR": "0",
                                "CARGO_MANIFEST_DIR": "$ROOT$.cargo/registry/src/github.com-1ecc6299db9ec823/libc-0.2.98",
                                "CARGO_PKG_VERSION": "0.2.98",
                                "CARGO_PKG_AUTHORS": "",
                                "CARGO_CRATE_NAME": "libc",
                                "CARGO_PKG_LICENSE_FILE": "",
                                "CARGO_PKG_HOMEPAGE": "",
                                "CARGO_PKG_DESCRIPTION": "",
                                "CARGO_PKG_NAME": "libc",
                                "CARGO_PKG_VERSION_PATCH": "98",
                                "CARGO": "cargo",
                                "CARGO_PKG_REPOSITORY": "",
                                "CARGO_PKG_VERSION_MINOR": "2",
                                "CARGO_PKG_VERSION_PRE": "",
                            },
                        },
                        dependencies: [
                            Dependency {
                                crate_id: CrateId(
                                    4,
                                ),
                                name: CrateName(
                                    "libc",
                                ),
                            },
                        ],
                        proc_macro: [],
                    },
                    CrateId(
                        2,
                    ): CrateData {
                        root_file_id: FileId(
                            3,
                        ),
                        edition: Edition2018,
                        display_name: Some(
                            CrateDisplayName {
                                crate_name: CrateName(
                                    "an_example",
                                ),
                                canonical_name: "an-example",
                            },
                        ),
                        cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "test",
                            ],
                        ),
                        potential_cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "test",
                            ],
                        ),
                        env: Env {
                            entries: {
                                "CARGO_PKG_LICENSE": "",
                                "CARGO_PKG_VERSION_MAJOR": "0",
                                "CARGO_MANIFEST_DIR": "$ROOT$hello-world",
                                "CARGO_PKG_VERSION": "0.1.0",
                                "CARGO_PKG_AUTHORS": "",
                                "CARGO_CRATE_NAME": "hello_world",
                                "CARGO_PKG_LICENSE_FILE": "",
                                "CARGO_PKG_HOMEPAGE": "",
                                "CARGO_PKG_DESCRIPTION": "",
                                "CARGO_PKG_NAME": "hello-world",
                                "CARGO_PKG_VERSION_PATCH": "0",
                                "CARGO": "cargo",
                                "CARGO_PKG_REPOSITORY": "",
                                "CARGO_PKG_VERSION_MINOR": "1",
                                "CARGO_PKG_VERSION_PRE": "",
                            },
                        },
                        dependencies: [
                            Dependency {
                                crate_id: CrateId(
                                    0,
                                ),
                                name: CrateName(
                                    "hello_world",
                                ),
                            },
                            Dependency {
                                crate_id: CrateId(
                                    4,
                                ),
                                name: CrateName(
                                    "libc",
                                ),
                            },
                        ],
                        proc_macro: [],
                    },
                    CrateId(
                        4,
                    ): CrateData {
                        root_file_id: FileId(
                            5,
                        ),
                        edition: Edition2015,
                        display_name: Some(
                            CrateDisplayName {
                                crate_name: CrateName(
                                    "libc",
                                ),
                                canonical_name: "libc",
                            },
                        ),
                        cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "feature=default",
                                "feature=std",
                                "test",
                            ],
                        ),
                        potential_cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "feature=align",
                                "feature=const-extern-fn",
                                "feature=default",
                                "feature=extra_traits",
                                "feature=rustc-dep-of-std",
                                "feature=std",
                                "feature=use_std",
                                "test",
                            ],
                        ),
                        env: Env {
                            entries: {
                                "CARGO_PKG_LICENSE": "",
                                "CARGO_PKG_VERSION_MAJOR": "0",
                                "CARGO_MANIFEST_DIR": "$ROOT$.cargo/registry/src/github.com-1ecc6299db9ec823/libc-0.2.98",
                                "CARGO_PKG_VERSION": "0.2.98",
                                "CARGO_PKG_AUTHORS": "",
                                "CARGO_CRATE_NAME": "libc",
                                "CARGO_PKG_LICENSE_FILE": "",
                                "CARGO_PKG_HOMEPAGE": "",
                                "CARGO_PKG_DESCRIPTION": "",
                                "CARGO_PKG_NAME": "libc",
                                "CARGO_PKG_VERSION_PATCH": "98",
                                "CARGO": "cargo",
                                "CARGO_PKG_REPOSITORY": "",
                                "CARGO_PKG_VERSION_MINOR": "2",
                                "CARGO_PKG_VERSION_PRE": "",
                            },
                        },
                        dependencies: [],
                        proc_macro: [],
                    },
                    CrateId(
                        1,
                    ): CrateData {
                        root_file_id: FileId(
                            2,
                        ),
                        edition: Edition2018,
                        display_name: Some(
                            CrateDisplayName {
                                crate_name: CrateName(
                                    "hello_world",
                                ),
                                canonical_name: "hello-world",
                            },
                        ),
                        cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "test",
                            ],
                        ),
                        potential_cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "test",
                            ],
                        ),
                        env: Env {
                            entries: {
                                "CARGO_PKG_LICENSE": "",
                                "CARGO_PKG_VERSION_MAJOR": "0",
                                "CARGO_MANIFEST_DIR": "$ROOT$hello-world",
                                "CARGO_PKG_VERSION": "0.1.0",
                                "CARGO_PKG_AUTHORS": "",
                                "CARGO_CRATE_NAME": "hello_world",
                                "CARGO_PKG_LICENSE_FILE": "",
                                "CARGO_PKG_HOMEPAGE": "",
                                "CARGO_PKG_DESCRIPTION": "",
                                "CARGO_PKG_NAME": "hello-world",
                                "CARGO_PKG_VERSION_PATCH": "0",
                                "CARGO": "cargo",
                                "CARGO_PKG_REPOSITORY": "",
                                "CARGO_PKG_VERSION_MINOR": "1",
                                "CARGO_PKG_VERSION_PRE": "",
                            },
                        },
                        dependencies: [
                            Dependency {
                                crate_id: CrateId(
                                    0,
                                ),
                                name: CrateName(
                                    "hello_world",
                                ),
                            },
                            Dependency {
                                crate_id: CrateId(
                                    4,
                                ),
                                name: CrateName(
                                    "libc",
                                ),
                            },
                        ],
                        proc_macro: [],
                    },
                    CrateId(
                        6,
                    ): CrateData {
                        root_file_id: FileId(
                            7,
                        ),
                        edition: Edition2015,
                        display_name: Some(
                            CrateDisplayName {
                                crate_name: CrateName(
                                    "build_script_build",
                                ),
                                canonical_name: "build-script-build",
                            },
                        ),
                        cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "feature=default",
                                "feature=std",
                                "test",
                            ],
                        ),
                        potential_cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "feature=align",
                                "feature=const-extern-fn",
                                "feature=default",
                                "feature=extra_traits",
                                "feature=rustc-dep-of-std",
                                "feature=std",
                                "feature=use_std",
                                "test",
                            ],
                        ),
                        env: Env {
                            entries: {
                                "CARGO_PKG_LICENSE": "",
                                "CARGO_PKG_VERSION_MAJOR": "0",
                                "CARGO_MANIFEST_DIR": "$ROOT$.cargo/registry/src/github.com-1ecc6299db9ec823/libc-0.2.98",
                                "CARGO_PKG_VERSION": "0.2.98",
                                "CARGO_PKG_AUTHORS": "",
                                "CARGO_CRATE_NAME": "libc",
                                "CARGO_PKG_LICENSE_FILE": "",
                                "CARGO_PKG_HOMEPAGE": "",
                                "CARGO_PKG_DESCRIPTION": "",
                                "CARGO_PKG_NAME": "libc",
                                "CARGO_PKG_VERSION_PATCH": "98",
                                "CARGO": "cargo",
                                "CARGO_PKG_REPOSITORY": "",
                                "CARGO_PKG_VERSION_MINOR": "2",
                                "CARGO_PKG_VERSION_PRE": "",
                            },
                        },
                        dependencies: [],
                        proc_macro: [],
                    },
                    CrateId(
                        3,
                    ): CrateData {
                        root_file_id: FileId(
                            4,
                        ),
                        edition: Edition2018,
                        display_name: Some(
                            CrateDisplayName {
                                crate_name: CrateName(
                                    "it",
                                ),
                                canonical_name: "it",
                            },
                        ),
                        cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "test",
                            ],
                        ),
                        potential_cfg_options: CfgOptions(
                            [
                                "debug_assertions",
                                "test",
                            ],
                        ),
                        env: Env {
                            entries: {
                                "CARGO_PKG_LICENSE": "",
                                "CARGO_PKG_VERSION_MAJOR": "0",
                                "CARGO_MANIFEST_DIR": "$ROOT$hello-world",
                                "CARGO_PKG_VERSION": "0.1.0",
                                "CARGO_PKG_AUTHORS": "",
                                "CARGO_CRATE_NAME": "hello_world",
                                "CARGO_PKG_LICENSE_FILE": "",
                                "CARGO_PKG_HOMEPAGE": "",
                                "CARGO_PKG_DESCRIPTION": "",
                                "CARGO_PKG_NAME": "hello-world",
                                "CARGO_PKG_VERSION_PATCH": "0",
                                "CARGO": "cargo",
                                "CARGO_PKG_REPOSITORY": "",
                                "CARGO_PKG_VERSION_MINOR": "1",
                                "CARGO_PKG_VERSION_PRE": "",
                            },
                        },
                        dependencies: [
                            Dependency {
                                crate_id: CrateId(
                                    0,
                                ),
                                name: CrateName(
                                    "hello_world",
                                ),
                            },
                            Dependency {
                                crate_id: CrateId(
                                    4,
                                ),
                                name: CrateName(
                                    "libc",
                                ),
                            },
                        ],
                        proc_macro: [],
                    },
                },
            }"#]],
    )
}
