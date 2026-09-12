struct Struct<'a> {
    string: &'a str,
}

fn struct_with_reserved_lifetime() {
    let _ = Struct<'_> {
        //~^ ERROR labels cannot use keyword names
        //~| ERROR expected `while`, `for`, `loop` or `{` after a label
        //~| ERROR comparison operators cannot be chained
        string: "",
    };
}

fn struct_with_named_lifetime() {
    let _ = Struct<'a> {
        //~^ ERROR expected `while`, `for`, `loop` or `{` after a label
        //~| ERROR comparison operators cannot be chained
        string: "",
    };
}

fn struct_with_multichar_lifetime() {
    let _ = Struct<'abc> {
        //~^ ERROR expected `while`, `for`, `loop` or `{` after a label
        //~| ERROR comparison operators cannot be chained
        string: "",
    };
}

fn main() {}
