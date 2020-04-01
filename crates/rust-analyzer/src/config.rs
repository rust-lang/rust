//! Config used by the language server.
//!
//! We currently get this config from `initialize` LSP request, which is not the
//! best way to do it, but was the simplest thing we could implement.
//!
//! Of particular interest is the `feature_flags` hash map: while other fields
//! configure the server itself, feature flags are passed into analysis, and
//! tweak things like automatic insertion of `()` in completions.

use rustc_hash::FxHashMap;

use crate::feature_flags::FeatureFlags;
use lsp_types::TextDocumentClientCapabilities;
use ra_flycheck::FlycheckConfig;
use ra_ide::{CompletionConfig, InlayHintsConfig};
use ra_project_model::CargoFeatures;
use serde::{Deserialize, Deserializer};

#[derive(Debug, Clone)]
pub struct Config {
    pub publish_decorations: bool,
    pub publish_diagnostics: bool,
    pub notifications: NotificationsConfig,
    pub supports_location_link: bool,
    pub line_folding_only: bool,
    pub inlay_hints: InlayHintsConfig,
    pub completion: CompletionConfig,
    pub call_info_full: bool,
    pub rustfmt: RustfmtConfig,
    pub check: Option<FlycheckConfig>,
    pub vscode_lldb: bool,
    pub proc_macro_srv: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NotificationsConfig {
    pub workspace_loaded: bool,
    pub cargo_toml_not_found: bool,
}

#[derive(Debug, Clone)]
pub enum RustfmtConfig {
    Rustfmt {
        extra_args: Vec<String>,
    },
    #[allow(unused)]
    CustomCommand {
        command: String,
        args: Vec<String>,
    },
}

impl Default for RustfmtConfig {
    fn default() -> Self {
        RustfmtConfig::Rustfmt { extra_args: Vec::new() }
    }
}

pub(crate) fn get_config(
    config: &ServerConfig,
    text_document_caps: Option<&TextDocumentClientCapabilities>,
) -> Config {
    let feature_flags = get_feature_flags(config);
    Config {
        publish_decorations: config.publish_decorations,
        publish_diagnostics: feature_flags.get("lsp.diagnostics"),
        notifications: NotificationsConfig {
            workspace_loaded: feature_flags.get("notifications.workspace-loaded"),
            cargo_toml_not_found: feature_flags.get("notifications.cargo-toml-not-found"),
        },
        supports_location_link: text_document_caps
            .and_then(|it| it.definition)
            .and_then(|it| it.link_support)
            .unwrap_or(false),
        line_folding_only: text_document_caps
            .and_then(|it| it.folding_range.as_ref())
            .and_then(|it| it.line_folding_only)
            .unwrap_or(false),
        inlay_hints: InlayHintsConfig {
            type_hints: config.inlay_hints_type,
            parameter_hints: config.inlay_hints_parameter,
            chaining_hints: config.inlay_hints_chaining,
            max_length: config.inlay_hints_max_length,
        },
        completion: CompletionConfig {
            enable_postfix_completions: feature_flags.get("completion.enable-postfix"),
            add_call_parenthesis: feature_flags.get("completion.insertion.add-call-parenthesis"),
            add_call_argument_snippets: feature_flags
                .get("completion.insertion.add-argument-snippets"),
        },
        call_info_full: feature_flags.get("call-info.full"),
        check: if config.cargo_watch_enable {
            Some(FlycheckConfig::CargoCommand {
                command: config.cargo_watch_command.clone(),
                all_targets: config.cargo_watch_all_targets,
                extra_args: config.cargo_watch_args.clone(),
            })
        } else {
            None
        },
        rustfmt: RustfmtConfig::Rustfmt { extra_args: config.rustfmt_args.clone() },
        vscode_lldb: config.vscode_lldb,
        proc_macro_srv: None, // FIXME: get this from config
    }
}

fn get_feature_flags(config: &ServerConfig) -> FeatureFlags {
    let mut ff = FeatureFlags::default();
    for (flag, &value) in &config.feature_flags {
        if ff.set(flag.as_str(), value).is_err() {
            log::error!("unknown feature flag: {:?}", flag);
        }
    }
    log::info!("feature_flags: {:#?}", ff);
    ff
}

/// Client provided initialization options
#[derive(Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase", default)]
pub struct ServerConfig {
    /// Whether the client supports our custom highlighting publishing decorations.
    /// This is different to the highlightingOn setting, which is whether the user
    /// wants our custom highlighting to be used.
    ///
    /// Defaults to `false`
    #[serde(deserialize_with = "nullable_bool_false")]
    pub publish_decorations: bool,

    pub exclude_globs: Vec<String>,
    #[serde(deserialize_with = "nullable_bool_false")]
    pub use_client_watching: bool,

    pub lru_capacity: Option<usize>,

    #[serde(deserialize_with = "nullable_bool_true")]
    pub inlay_hints_type: bool,
    #[serde(deserialize_with = "nullable_bool_true")]
    pub inlay_hints_parameter: bool,
    #[serde(deserialize_with = "nullable_bool_true")]
    pub inlay_hints_chaining: bool,
    pub inlay_hints_max_length: Option<usize>,

    pub cargo_watch_enable: bool,
    pub cargo_watch_args: Vec<String>,
    pub cargo_watch_command: String,
    pub cargo_watch_all_targets: bool,

    /// For internal usage to make integrated tests faster.
    #[serde(deserialize_with = "nullable_bool_true")]
    pub with_sysroot: bool,

    /// Fine grained feature flags to disable specific features.
    pub feature_flags: FxHashMap<String, bool>,

    pub rustfmt_args: Vec<String>,

    /// Cargo feature configurations.
    pub cargo_features: CargoFeatures,

    /// Enabled if the vscode_lldb extension is available.
    pub vscode_lldb: bool,
}

impl Default for ServerConfig {
    fn default() -> ServerConfig {
        ServerConfig {
            publish_decorations: false,
            exclude_globs: Vec::new(),
            use_client_watching: false,
            lru_capacity: None,
            inlay_hints_type: true,
            inlay_hints_parameter: true,
            inlay_hints_chaining: true,
            inlay_hints_max_length: None,
            cargo_watch_enable: true,
            cargo_watch_args: Vec::new(),
            cargo_watch_command: "check".to_string(),
            cargo_watch_all_targets: true,
            with_sysroot: true,
            feature_flags: FxHashMap::default(),
            cargo_features: Default::default(),
            rustfmt_args: Vec::new(),
            vscode_lldb: false,
        }
    }
}

/// Deserializes a null value to a bool false by default
fn nullable_bool_false<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::deserialize(deserializer)?;
    Ok(opt.unwrap_or(false))
}

/// Deserializes a null value to a bool true by default
fn nullable_bool_true<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::deserialize(deserializer)?;
    Ok(opt.unwrap_or(true))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn deserialize_init_options_defaults() {
        // check that null == default for both fields
        let default = ServerConfig::default();
        assert_eq!(default, serde_json::from_str(r#"{}"#).unwrap());
        assert_eq!(
            default,
            serde_json::from_str(r#"{"publishDecorations":null, "lruCapacity":null}"#).unwrap()
        );
    }
}
