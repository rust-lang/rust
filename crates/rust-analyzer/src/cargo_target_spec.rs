//! See `CargoTargetSpec`

use std::mem;

use cfg::{CfgAtom, CfgExpr};
use ide::{FileId, RunnableKind, TestId};
use project_model::{self, ManifestPath, TargetKind};
use vfs::AbsPathBuf;

use crate::{global_state::GlobalStateSnapshot, Result};

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
    pub(crate) required_features: Vec<String>,
}

impl CargoTargetSpec {
    pub(crate) fn runnable_args(
        snap: &GlobalStateSnapshot,
        mut spec: Option<CargoTargetSpec>,
        kind: &RunnableKind,
        cfg: &Option<CfgExpr>,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let mut args = Vec::new();
        let mut extra_args = Vec::new();

        let target_required_features =
            spec.as_mut().map(|spec| mem::take(&mut spec.required_features)).unwrap_or(Vec::new());

        match kind {
            RunnableKind::Test { test_id, attr } => {
                args.push("test".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut args, kind);
                }
                extra_args.push(test_id.to_string());
                if let TestId::Path(_) = test_id {
                    extra_args.push("--exact".to_string());
                }
                extra_args.push("--nocapture".to_string());
                if attr.ignore {
                    extra_args.push("--ignored".to_string());
                }
            }
            RunnableKind::TestMod { path } => {
                args.push("test".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut args, kind);
                }
                extra_args.push(path.to_string());
                extra_args.push("--nocapture".to_string());
            }
            RunnableKind::Bench { test_id } => {
                args.push("bench".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut args, kind);
                }
                extra_args.push(test_id.to_string());
                if let TestId::Path(_) = test_id {
                    extra_args.push("--exact".to_string());
                }
                extra_args.push("--nocapture".to_string());
            }
            RunnableKind::DocTest { test_id } => {
                args.push("test".to_string());
                args.push("--doc".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut args, kind);
                }
                extra_args.push(test_id.to_string());
                extra_args.push("--nocapture".to_string());
            }
            RunnableKind::Bin => {
                let subcommand = match spec {
                    Some(CargoTargetSpec { target_kind: TargetKind::Test, .. }) => "test",
                    _ => "run",
                };
                args.push(subcommand.to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut args, kind);
                }
            }
        }

        let cargo_config = snap.config.cargo();
        if cargo_config.all_features {
            args.push("--all-features".to_string());

            for feature in target_required_features {
                args.push("--features".to_string());
                args.push(feature);
            }
        } else {
            let mut features = Vec::new();
            if let Some(cfg) = cfg.as_ref() {
                required_features(cfg, &mut features);
            }

            features.extend(cargo_config.features);
            features.extend(target_required_features);

            features.dedup();
            for feature in features {
                args.push("--features".to_string());
                args.push(feature);
            }
        }

        Ok((args, extra_args))
    }

    pub(crate) fn for_file(
        global_state_snapshot: &GlobalStateSnapshot,
        file_id: FileId,
    ) -> Result<Option<CargoTargetSpec>> {
        let crate_id = match global_state_snapshot.analysis.crate_for(file_id)?.first() {
            Some(crate_id) => *crate_id,
            None => return Ok(None),
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
        };

        Ok(Some(res))
    }

    pub(crate) fn push_to(self, buf: &mut Vec<String>, kind: &RunnableKind) {
        buf.push("--package".to_string());
        buf.push(self.package);

        // Can't mix --doc with other target flags
        if let RunnableKind::DocTest { .. } = kind {
            return;
        }
        match self.target_kind {
            TargetKind::Bin => {
                buf.push("--bin".to_string());
                buf.push(self.target);
            }
            TargetKind::Test => {
                buf.push("--test".to_string());
                buf.push(self.target);
            }
            TargetKind::Bench => {
                buf.push("--bench".to_string());
                buf.push(self.target);
            }
            TargetKind::Example => {
                buf.push("--example".to_string());
                buf.push(self.target);
            }
            TargetKind::Lib => {
                buf.push("--lib".to_string());
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
