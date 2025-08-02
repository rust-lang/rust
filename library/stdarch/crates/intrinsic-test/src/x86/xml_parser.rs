use serde::{Deserialize, Deserializer};


// Custom deserializer function to convert strings to u32
fn string_to_u32<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    return s.as_str().parse::<u32>().or(Ok(0u32));
}

#[derive(Deserialize)]
struct Data {
    #[serde(rename = "intrinsic", default)]
    intrinsics: Vec<XMLIntrinsic>,
}

#[derive(Deserialize)]
struct XMLIntrinsic {
    #[serde(rename = "return")]
    return_data: Parameter,
    #[serde(rename = "@name")]
    name: String,
    // #[serde(rename = "@tech")]
    // tech: String,
    #[serde(rename = "CPUID", default)]
    cpuid: Vec<String>,
    #[serde(rename = "parameter", default)]
    parameters: Vec<Parameter>,
}

#[derive(Deserialize)]
pub struct Parameter {
    #[serde(rename = "@varname")]
    pub var_name: String,
    #[serde(rename = "@type")]
    pub type_data: String,
    #[serde(rename = "@etype", default)]
    pub etype: String,
    #[serde(rename = "@memwidth", default, deserialize_with = "string_to_u32")]
    pub memwidth: u32,
    #[serde(rename = "@immtype", default)]
    pub imm_type: String,
}
