enum Enum {
    Value(String),
    Other,
}

fn main() {
    let v = vec![Enum::Value("str".to_owned()), Enum::Other];
    let _ = v.iter().filter(|t| matches!(t, Enum::Value("str"))).count();
    //~^ ERROR mismatched types

    match Enum::Other {
        Enum::Value("str") => {}
        //~^ ERROR mismatched types
        _ => {}
    }

    let pair = (String::new(), 1);
    if let ("a", 1) = pair {}
    //~^ ERROR mismatched types

    let opt = Some(String::new());
    if let Some("a") = opt {}
    //~^ ERROR mismatched types
}
