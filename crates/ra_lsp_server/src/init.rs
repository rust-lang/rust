use serde::{Deserialize, Deserializer};

/// Client provided initialization options
#[derive(Deserialize, Clone, Copy, Debug)]
#[serde(rename_all = "camelCase")]
pub struct InitializationOptions {
    /// Whether the client supports our custom highlighting publishing decorations.
    /// This is different to the highlightingOn setting, which is whether the user
    /// wants our custom highlighting to be used.
    ///
    /// Defaults to `true`
    #[serde(default = "bool_true", deserialize_with = "nullable_bool_true")]
    pub publish_decorations: bool,

    /// Whether or not the workspace loaded notification should be sent
    ///
    /// Defaults to `true`
    #[serde(default = "bool_true", deserialize_with = "nullable_bool_true")]
    pub show_workspace_loaded: bool,
}

impl Default for InitializationOptions {
    fn default() -> InitializationOptions {
        InitializationOptions { publish_decorations: true, show_workspace_loaded: true }
    }
}

fn bool_true() -> bool {
    true
}

/// Deserializes a null value to a bool true by default
fn nullable_bool_true<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::deserialize(deserializer)?;
    Ok(opt.unwrap_or(true))
}
