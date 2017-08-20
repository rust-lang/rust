// rustfmt-attributes_on_same_line_as_variant: true
// Option to place attributes on the same line as variants where possible

enum Lorem {
    #[ serde(skip_serializing) ]
    Ipsum,
    #[ serde(skip_serializing) ]
    Dolor,
    #[ serde(skip_serializing) ]
    Amet,
}
