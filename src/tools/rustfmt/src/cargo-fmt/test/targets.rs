use super::*;

struct ExpTarget {
    path: &'static str,
    edition: &'static str,
    kind: &'static str,
}

mod all_targets {
    use super::*;

    fn assert_correct_targets_loaded(
        manifest_suffix: &str,
        source_root: &str,
        exp_targets: &[ExpTarget],
        exp_num_targets: usize,
    ) {
        let root_path = Path::new("tests/cargo-fmt/source").join(source_root);
        let get_path = |exp: &str| PathBuf::from(&root_path).join(exp).canonicalize().unwrap();
        let manifest_path = Path::new(&root_path).join(manifest_suffix);
        let targets = get_targets(&CargoFmtStrategy::All, Some(manifest_path.as_path()))
            .expect("Targets should have been loaded");

        assert_eq!(targets.len(), exp_num_targets);

        for target in exp_targets {
            assert!(targets.contains(&Target {
                path: get_path(target.path),
                edition: target.edition.to_owned(),
                kind: target.kind.to_owned(),
            }));
        }
    }

    mod different_crate_and_dir_names {
        use super::*;

        fn assert_correct_targets_loaded(manifest_suffix: &str) {
            let exp_targets = vec![
                ExpTarget {
                    path: "dependency-dir-name/subdep-dir-name/src/lib.rs",
                    edition: "2018",
                    kind: "lib",
                },
                ExpTarget {
                    path: "dependency-dir-name/src/lib.rs",
                    edition: "2018",
                    kind: "lib",
                },
                ExpTarget {
                    path: "src/main.rs",
                    edition: "2018",
                    kind: "main",
                },
            ];
            super::assert_correct_targets_loaded(
                manifest_suffix,
                "divergent-crate-dir-names",
                &exp_targets,
                3,
            );
        }

        #[test]
        fn correct_targets_from_root() {
            assert_correct_targets_loaded("Cargo.toml");
        }

        #[test]
        fn correct_targets_from_sub_local_dep() {
            assert_correct_targets_loaded("dependency-dir-name/Cargo.toml");
        }
    }

    mod workspaces {
        use super::*;

        fn assert_correct_targets_loaded(manifest_suffix: &str) {
            let exp_targets = vec![
                ExpTarget {
                    path: "ws/a/src/main.rs",
                    edition: "2018",
                    kind: "bin",
                },
                ExpTarget {
                    path: "ws/b/src/main.rs",
                    edition: "2018",
                    kind: "bin",
                },
                ExpTarget {
                    path: "ws/c/src/lib.rs",
                    edition: "2018",
                    kind: "lib",
                },
                ExpTarget {
                    path: "ws/a/d/src/lib.rs",
                    edition: "2018",
                    kind: "lib",
                },
                ExpTarget {
                    path: "e/src/main.rs",
                    edition: "2018",
                    kind: "main",
                },
                ExpTarget {
                    path: "ws/a/d/f/src/lib.rs",
                    edition: "2018",
                    kind: "lib",
                },
            ];
            super::assert_correct_targets_loaded(
                manifest_suffix,
                "workspaces/path-dep-above",
                &exp_targets,
                6,
            );
        }

        #[test]
        fn includes_outside_workspace_deps() {
            assert_correct_targets_loaded("ws/Cargo.toml");
        }

        #[test]
        fn includes_workspace_from_dep_above() {
            assert_correct_targets_loaded("e/Cargo.toml");
        }

        #[test]
        fn includes_all_packages_from_workspace_subdir() {
            assert_correct_targets_loaded("ws/a/d/f/Cargo.toml");
        }
    }
}
