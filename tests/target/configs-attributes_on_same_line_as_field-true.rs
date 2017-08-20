// rustfmt-attributes_on_same_line_as_field: true
// Option to place attributes on the same line as fields where possible

struct Lorem {
    #[serde(rename = "Ipsum")] ipsum: usize,
    #[serde(rename = "Dolor")] dolor: usize,
    #[serde(rename = "Amet")] amet: usize,
}
