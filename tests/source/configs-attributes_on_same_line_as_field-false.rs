// rustfmt-attributes_on_same_line_as_field: false
// Option to place attributes on the same line as fields where possible

struct Lorem {
    #[ serde(rename = "Ipsum") ]
    ipsum: usize,
    #[ serde(rename = "Dolor") ]
    dolor: usize,
    #[ serde(rename = "Amet") ]
    amet: usize,
}

// #1943
pub struct Bzip2 {
    # [ serde (rename = "level") ]
     level: i32 ,
}
