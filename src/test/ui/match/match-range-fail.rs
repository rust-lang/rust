fn main() {
    match "wow" {
        "bar" ..= "foo" => { }
    };
    //~^^ ERROR only char and numeric types are allowed in range
    //~| start type: &'static str
    //~| end type: &'static str

    match "wow" {
        10 ..= "what" => ()
    };
    //~^^ ERROR only char and numeric types are allowed in range
    //~| start type: {integer}
    //~| end type: &'static str

    match 5 {
        'c' ..= 100 => { }
        _ => { }
    };
    //~^^^ ERROR mismatched types
    //~| expected type `{integer}`
    //~| found type `char`
}
