//! See `CargoTargetSpec`

use std::mem;

use cfg::{CfgAtom, CfgExpr};
use ide::{Cancellable, CrateId, FileId, RunnableKind, TestId};
use project_model::{self, CargoFeatures, ManifestPath, TargetKind};
use rustc_hash::FxHashSet;
use vfs::AbsPathBuf;

use crate::global_state::GlobalStateSnapshot;

/// Abstract representation of Cargo target.
///
/// We use it to cook up the set of cli args we need to pass to Cargo to
/// build/test/run the target.
#[derive(Clone)]
pub(crate) struct CargoTargetSpec {
    pub(crate) workspace_root: AbsPathBuf,
    pub(crate) cargo_toml: ManifestPath,
    pub(crate) package: String,
    pub(crate) target: String,
    pub(crate) target_kind: TargetKind,
    pub(crate) crate_id: CrateId,
    pub(crate) required_features: Vec<String>,
    pub(crate) features: FxHashSet<String>,
}

impl CargoTargetSpec {
    pub(crate) fn runnable_args(
        snap: &GlobalStateSnapshot,
        spec: Option<CargoTargetSpec>,
        kind: &RunnableKind,
        cfg: &Option<CfgExpr>,
    ) -> (Vec<String>, Vec<String>) {
        let mut args = Vec::new();
        let mut extra_args = Vec::new();

        match kind {
            RunnableKind::Test { test_id, attr } => {
                args.push("test".to_owned());
                extra_args.push(test_id.to_string());
                if let TestId::Path(_) = test_id {
                    extra_args.push("--exact".to_owned());
                }
                extra_args.push("--nocapture".to_owned());
                if attr.ignore {
                    extra_args.push("--ignored".to_owned());
                }
            }
            RunnableKind::TestMod { path } => {
                args.push("test".to_owned());
                extra_args.push(path.clone());
                extra_args.push("--nocapture".to_owned());
            }
            RunnableKind::Bench { test_id } => {
                args.push("bench".to_owned());
                extra_args.push(test_id.to_string());
                if let TestId::Path(_) = test_id {
                    extra_args.push("--exact".to_owned());
                }
                extra_args.push("--nocapture".to_owned());
            }
            RunnableKind::DocTest { test_id } => {
                args.push("test".to_owned());
                args.push("--doc".to_owned());
                extra_args.push(test_id.to_string());
                extra_args.push("--nocapture".to_owned());
            }
            RunnableKind::Bin => {
                let subcommand = match spec {
                    Some(CargoTargetSpec { target_kind: TargetKind::Test, .. }) => "test",
                    _ => "run",
                };
                args.push(subcommand.to_owned());
            }
        }

        let (allowed_features, target_required_features) = if let Some(mut spec) = spec {
            let allowed_features = mem::take(&mut spec.features);
            let required_features = mem::take(&mut spec.required_features);
            spec.push_to(&mut args, kind);
            (allowed_features, required_features)
        } else {
            (Default::default(), Default::default())
        };

        let cargo_config = snap.config.cargo();

        match &cargo_config.features {
            CargoFeatures::All => {
                args.push("--all-features".to_owned());
                for feature in target_required_features {
                    args.push("--features".to_owned());
                    args.push(feature);
                }
            }
            CargoFeatures::Selected { features, no_default_features } => {
                let mut feats = Vec::new();
                if let Some(cfg) = cfg.as_ref() {
                    required_features(cfg, &mut feats);
                }

                feats.extend(
                    features.iter().filter(|&feat| allowed_features.contains(feat)).cloned(),
                );
                feats.extend(target_required_features);

                feats.dedup();
                for feature in feats {
                    args.push("--features".to_owned());
                    args.push(feature);
                }

                if *no_default_features {
                    args.push("--no-default-features".to_owned());
                }
            }
        }
        (args, extra_args)
    }

    pub(crate) fn for_file(
        global_state_snapshot: &GlobalStateSnapshot,
        file_id: FileId,
    ) -> Cancellable<Option<CargoTargetSpec>> {
        let crate_id = match &*global_state_snapshot.analysis.crates_for(file_id)? {
            &[crate_id, ..] => crate_id,
            _ => return Ok(None),
        };
        let (cargo_ws, target) = match global_state_snapshot.cargo_target_for_crate_root(crate_id) {
            Some(it) => it,
            None => return Ok(None),
        };

        let target_data = &cargo_ws[target];
        let package_data = &cargo_ws[target_data.package];
        let res = CargoTargetSpec {
            workspace_root: cargo_ws.workspace_root().to_path_buf(),
            cargo_toml: package_data.manifest.clone(),
            package: cargo_ws.package_flag(package_data),
            target: target_data.name.clone(),
            target_kind: target_data.kind,
            required_features: target_data.required_features.clone(),
            features: package_data.features.keys().cloned().collect(),
            crate_id,
        };

        Ok(Some(res))
    }

    pub(crate) fn push_to(self, buf: &mut Vec<String>, kind: &RunnableKind) {
        buf.push("--package".to_owned());
        buf.push(self.package);

        // Can't mix --doc with other target flags
        if let RunnableKind::DocTest { .. } = kind {
            return;
        }
        match self.target_kind {
            TargetKind::Bin => {
                buf.push("--bin".to_owned());
                buf.push(self.target);
            }
            TargetKind::Test => {
                buf.push("--test".to_owned());
                buf.push(self.target);
            }
            TargetKind::Bench => {
                buf.push("--bench".to_owned());
                buf.push(self.target);
            }
            TargetKind::Example => {
                buf.push("--example".to_owned());
                buf.push(self.target);
            }
            TargetKind::Lib => {
                buf.push("--lib".to_owned());
            }
            TargetKind::Other | TargetKind::BuildScript => (),
        }
    }
}

/// Fill minimal features needed
fn required_features(cfg_expr: &CfgExpr, features: &mut Vec<String>) {
    match cfg_expr {
        CfgExpr::Atom(CfgAtom::KeyValue { key, value }) if key == "feature" => {
            features.push(value.to_string())
        }
        CfgExpr::All(preds) => {
            preds.iter().for_each(|cfg| required_features(cfg, features));
        }
        CfgExpr::Any(preds) => {
            for cfg in preds {
                let len_features = features.len();
                required_features(cfg, features);
                if len_features != features.len() {
                    break;
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use cfg::CfgExpr;
    use mbe::syntax_node_to_token_tree;
    use syntax::{
        ast::{self, AstNode},
        SmolStr,
    };

    fn check(cfg: &str, expected_features: &[&str]) {
        let cfg_expr = {
            let source_file = ast::SourceFile::parse(cfg).ok().unwrap();
            let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
            let (tt, _) = syntax_node_to_token_tree(tt.syntax());
            CfgExpr::parse(&tt)
        };

        let mut features = vec![];
        required_features(&cfg_expr, &mut features);

        let expected_features =
            expected_features.iter().map(|&it| SmolStr::new(it)).collect::<Vec<_>>();

        assert_eq!(features, expected_features);
    }

    #[test]
    fn test_cfg_expr_minimal_features_needed() {
        check(r#"#![cfg(feature = "baz")]"#, &["baz"]);
        check(r#"#![cfg(all(feature = "baz", feature = "foo"))]"#, &["baz", "foo"]);
        check(r#"#![cfg(any(feature = "baz", feature = "foo", unix))]"#, &["baz"]);
        check(r#"#![cfg(foo)]"#, &[]);
    }
}
