// rustfmt-same_line_attributes: true
// Option to place attributes on the same line as fields and variants where possible

enum Lorem {
    #[serde(skip_serializing)] Ipsum,
    #[serde(skip_serializing)] Dolor,
    #[serde(skip_serializing)] Amet,
}

struct Lorem {
    #[serde(rename = "Ipsum")] ipsum: usize,
    #[serde(rename = "Dolor")] dolor: usize,
    #[serde(rename = "Amet")] amet: usize,
}
