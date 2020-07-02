//! See `CargoTargetSpec`

use ra_cfg::CfgExpr;
use ra_db::AbsPathBuf;
use ra_ide::{FileId, RunnableKind, TestId};
use ra_project_model::{self, TargetKind};

use crate::{global_state::GlobalStateSnapshot, Result};

/// Abstract representation of Cargo target.
///
/// We use it to cook up the set of cli args we need to pass to Cargo to
/// build/test/run the target.
#[derive(Clone)]
pub(crate) struct CargoTargetSpec {
    pub(crate) workspace_root: AbsPathBuf,
    pub(crate) package: String,
    pub(crate) target: String,
    pub(crate) target_kind: TargetKind,
}

impl CargoTargetSpec {
    pub(crate) fn runnable_args(
        snap: &GlobalStateSnapshot,
        spec: Option<CargoTargetSpec>,
        kind: &RunnableKind,
        cfgs: &[CfgExpr],
    ) -> Result<(Vec<String>, Vec<String>)> {
        let mut args = Vec::new();
        let mut extra_args = Vec::new();
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
                args.push("run".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut args, kind);
                }
            }
        }

        if snap.config.cargo.all_features {
            args.push("--all-features".to_string());
        } else {
            let mut features = Vec::new();
            for cfg in cfgs {
                required_features(cfg, &mut features);
            }
            for feature in &snap.config.cargo.features {
                features.push(feature.clone());
            }
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
        let res = CargoTargetSpec {
            workspace_root: cargo_ws.workspace_root().to_path_buf(),
            package: cargo_ws.package_flag(&cargo_ws[cargo_ws[target].package]),
            target: cargo_ws[target].name.clone(),
            target_kind: cargo_ws[target].kind,
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
            TargetKind::Other => (),
        }
    }
}

/// Fill minimal features needed
fn required_features(cfg_expr: &CfgExpr, features: &mut Vec<String>) {
    match cfg_expr {
        CfgExpr::KeyValue { key, value } if key == "feature" => features.push(value.to_string()),
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

    use mbe::{ast_to_token_tree, TokenMap};
    use ra_cfg::parse_cfg;
    use ra_syntax::{
        ast::{self, AstNode},
        SmolStr,
    };

    fn get_token_tree_generated(input: &str) -> (tt::Subtree, TokenMap) {
        let source_file = ast::SourceFile::parse(input).ok().unwrap();
        let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
        ast_to_token_tree(&tt).unwrap()
    }

    #[test]
    fn test_cfg_expr_minimal_features_needed() {
        let (subtree, _) = get_token_tree_generated(r#"#![cfg(feature = "baz")]"#);
        let cfg_expr = parse_cfg(&subtree);
        let mut min_features = vec![];
        required_features(&cfg_expr, &mut min_features);

        assert_eq!(min_features, vec![SmolStr::new("baz")]);

        let (subtree, _) =
            get_token_tree_generated(r#"#![cfg(all(feature = "baz", feature = "foo"))]"#);
        let cfg_expr = parse_cfg(&subtree);

        let mut min_features = vec![];
        required_features(&cfg_expr, &mut min_features);
        assert_eq!(min_features, vec![SmolStr::new("baz"), SmolStr::new("foo")]);

        let (subtree, _) =
            get_token_tree_generated(r#"#![cfg(any(feature = "baz", feature = "foo", unix))]"#);
        let cfg_expr = parse_cfg(&subtree);

        let mut min_features = vec![];
        required_features(&cfg_expr, &mut min_features);
        assert_eq!(min_features, vec![SmolStr::new("baz")]);

        let (subtree, _) = get_token_tree_generated(r#"#![cfg(foo)]"#);
        let cfg_expr = parse_cfg(&subtree);

        let mut min_features = vec![];
        required_features(&cfg_expr, &mut min_features);
        assert!(min_features.is_empty());
    }
}
