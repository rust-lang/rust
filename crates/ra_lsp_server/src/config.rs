use serde::{Deserialize, Deserializer};

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

    /// Whether or not the workspace loaded notification should be sent
    ///
    /// Defaults to `true`
    #[serde(deserialize_with = "nullable_bool_true")]
    pub show_workspace_loaded: bool,

    pub exclude_globs: Vec<String>,

    pub lru_capacity: Option<usize>,
}

impl Default for ServerConfig {
    fn default() -> ServerConfig {
        ServerConfig {
            publish_decorations: false,
            show_workspace_loaded: true,
            exclude_globs: Vec::new(),
            lru_capacity: None,
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
            serde_json::from_str(
                r#"{"publishDecorations":null, "showWorkspaceLoaded":null, "lruCapacity":null}"#
            )
            .unwrap()
        );
    }
}
