//! See [`patch_json_for_outdated_configs`]
use serde_json::{Value, json};

/// This function patches the json config to the new expected keys.
/// That is we try to load old known config keys here and convert them to the new ones.
/// See https://github.com/rust-lang/rust-analyzer/pull/12010
///
/// We already have an alias system for simple cases, but if we make structural changes
/// the alias infra fails down.
pub(super) fn patch_json_for_outdated_configs(json: &mut Value) {
    let copy = json.clone();

    macro_rules! patch {
        ($(
            $($src:ident).+ -> $($dst:ident).+ ;
        )+) => { $(
            match copy.pointer(concat!($("/", stringify!($src)),+)).cloned() {
                Some(Value::Object(_)) | None => (),
                Some(it) => {
                    let mut last = it;
                    for segment in [$(stringify!($dst)),+].into_iter().rev() {
                        last = Value::Object(serde_json::Map::from_iter(std::iter::once((segment.to_owned(), last))));
                    }

                    merge(json, last);
                },
            }
        )+ };
    }

    patch! {
        assist.allowMergingIntoGlobImports -> imports.merge.glob;
        assist.exprFillDefault -> assist.expressionFillDefault;
        assist.importEnforceGranularity -> imports.granularity.enforce;
        assist.importGranularity -> imports.granularity.group;
        assist.importMergeBehavior -> imports.granularity.group;
        assist.importMergeBehaviour -> imports.granularity.group;
        assist.importGroup -> imports.group.enable;
        assist.importPrefix -> imports.prefix;
        primeCaches.enable -> cachePriming.enable;
        cache.warmup -> cachePriming.enable;
        cargo.loadOutDirsFromCheck -> cargo.buildScripts.enable;
        cargo.runBuildScripts -> cargo.buildScripts.enable;
        cargo.runBuildScriptsCommand -> cargo.buildScripts.overrideCommand;
        cargo.useRustcWrapperForBuildScripts -> cargo.buildScripts.useRustcWrapper;
        diagnostics.enableExperimental -> diagnostics.experimental.enable;
        experimental.procAttrMacros -> procMacro.attributes.enable;
        highlighting.strings -> semanticHighlighting.strings.enable;
        highlightRelated.breakPoints -> semanticHighlighting.breakPoints.enable;
        highlightRelated.exitPoints -> semanticHighlighting.exitPoints.enable;
        highlightRelated.yieldPoints -> semanticHighlighting.yieldPoints.enable;
        highlightRelated.references -> semanticHighlighting.references.enable;
        hover.documentation -> hover.documentation.enable;
        hover.linksInHover -> hover.links.enable;
        hoverActions.linksInHover -> hover.links.enable;
        hoverActions.debug -> hover.actions.debug.enable;
        hoverActions.enable -> hover.actions.enable;
        hoverActions.gotoTypeDef -> hover.actions.gotoTypeDef.enable;
        hoverActions.implementations -> hover.actions.implementations.enable;
        hoverActions.references -> hover.actions.references.enable;
        hoverActions.run -> hover.actions.run.enable;
        inlayHints.chainingHints -> inlayHints.chainingHints.enable;
        inlayHints.closureReturnTypeHints -> inlayHints.closureReturnTypeHints.enable;
        inlayHints.hideNamedConstructorHints -> inlayHints.typeHints.hideNamedConstructorHints;
        inlayHints.parameterHints -> inlayHints.parameterHints.enable;
        inlayHints.reborrowHints -> inlayHints.reborrowHints.enable;
        inlayHints.typeHints -> inlayHints.typeHints.enable;
        lruCapacity -> lru.capacity;
        runnables.cargoExtraArgs -> runnables.extraArgs ;
        runnables.overrideCargo -> runnables.command ;
        rustcSource -> rustc.source;
        rustfmt.enableRangeFormatting -> rustfmt.rangeFormatting.enable;
    }

    // completion.snippets -> completion.snippets.custom;
    if let Some(Value::Object(obj)) = copy.pointer("/completion/snippets").cloned()
        && (obj.len() != 1 || obj.get("custom").is_none())
    {
        merge(
            json,
            json! {{
                "completion": {
                    "snippets": {
                        "custom": obj
                    },
                },
            }},
        );
    }

    // callInfo_full -> signatureInfo_detail, signatureInfo_documentation_enable
    if let Some(Value::Bool(b)) = copy.pointer("/callInfo/full") {
        let sig_info = match b {
            true => json!({ "signatureInfo": {
                "documentation": {"enable": true}},
                "detail": "full"
            }),
            false => json!({ "signatureInfo": {
                "documentation": {"enable": false}},
                "detail": "parameters"
            }),
        };
        merge(json, sig_info);
    }

    // cargo_allFeatures, cargo_features -> cargo_features
    if let Some(Value::Bool(true)) = copy.pointer("/cargo/allFeatures") {
        merge(json, json!({ "cargo": { "features": "all" } }));
    }

    // checkOnSave_allFeatures, checkOnSave_features -> check_features
    if let Some(Value::Bool(true)) = copy.pointer("/checkOnSave/allFeatures") {
        merge(json, json!({ "check": { "features": "all" } }));
    }

    // completion_addCallArgumentSnippets completion_addCallParenthesis -> completion_callable_snippets
    'completion: {
        let res = match (
            copy.pointer("/completion/addCallArgumentSnippets"),
            copy.pointer("/completion/addCallParenthesis"),
        ) {
            (Some(Value::Bool(true)), Some(Value::Bool(true))) => json!("fill_arguments"),
            (_, Some(Value::Bool(true))) => json!("add_parentheses"),
            (Some(Value::Bool(false)), Some(Value::Bool(false))) => json!("none"),
            (_, _) => break 'completion,
        };
        merge(json, json!({ "completion": { "callable": {"snippets": res }} }));
    }

    // We need to do this due to the checkOnSave_enable -> checkOnSave change, as that key now can either be an object or a bool
    // checkOnSave_* -> check_*
    if let Some(Value::Object(obj)) = copy.pointer("/checkOnSave") {
        // checkOnSave_enable -> checkOnSave
        if let Some(b @ Value::Bool(_)) = obj.get("enable") {
            merge(json, json!({ "checkOnSave": b }));
        }
        merge(json, json!({ "check": obj }));
    }
}

fn merge(dst: &mut Value, src: Value) {
    match (dst, src) {
        (Value::Object(dst), Value::Object(src)) => {
            for (k, v) in src {
                merge(dst.entry(k).or_insert(v.clone()), v)
            }
        }
        (dst, src) => *dst = src,
    }
}

#[test]
fn check_on_save_patching() {
    let mut json = json!({ "checkOnSave": { "overrideCommand": "foo" }});
    patch_json_for_outdated_configs(&mut json);
    assert_eq!(
        json,
        json!({ "checkOnSave": { "overrideCommand": "foo" }, "check": { "overrideCommand": "foo" }})
    );
}

#[test]
fn check_on_save_patching_enable() {
    let mut json = json!({ "checkOnSave": { "enable": true, "overrideCommand": "foo" }});
    patch_json_for_outdated_configs(&mut json);
    assert_eq!(
        json,
        json!({ "checkOnSave": true, "check": { "enable": true, "overrideCommand": "foo" }})
    );
}
