struct Struct<'a> {
    string: &'a str,
}

fn struct_with_reserved_lifetime() {
    let _ = Struct<'_> {
        //~^ ERROR labels cannot use keyword names
        string: "",
    };
}

fn struct_with_named_lifetime<'a>() {
    let _ = Struct<'a> {
        //~^ ERROR expected `while`, `for`, `loop` or `{` after a label
        string: "",
    };
}

fn struct_with_multichar_lifetime<'abc>() {
    let _ = Struct<'abc> {
        //~^ ERROR expected `while`, `for`, `loop` or `{` after a label
        string: "",
    };
}

fn main() {}
