use serde::Serialize;

pub(crate) const CURRENT_SCHEMA_VERSION: usize = 1;

#[derive(Serialize)]
pub(crate) struct Schema {
    pub(crate) schema_version: usize,
    pub(crate) items: Vec<SchemaItem>,
}

#[derive(Serialize)]
pub(crate) struct SchemaItem {
    pub(crate) name: String,
    pub(crate) url: Option<String>,
    pub(crate) deprecated: bool,
    pub(crate) children: Vec<SchemaItem>,
}
