fn main() {
    match "wow" {
        "bar" ..= "foo" => { }
    };
    //~^^ ERROR only `char` and numeric types are allowed in range

    match "wow" {
        10 ..= "what" => ()
    };
    //~^^ ERROR only `char` and numeric types are allowed in range

    match "wow" {
        true ..= "what" => {}
    };
    //~^^ ERROR only `char` and numeric types are allowed in range

    match 5 {
        'c' ..= 100 => { }
        _ => { }
    };
    //~^^^ ERROR mismatched types
}
