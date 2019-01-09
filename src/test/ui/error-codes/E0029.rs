fn main() {
    let s = "hoho";

    match s {
        "hello" ..= "world" => {}
        //~^ ERROR only char and numeric types are allowed in range patterns
        _ => {}
    }
}
