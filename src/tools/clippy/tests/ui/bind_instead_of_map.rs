#![deny(clippy::bind_instead_of_map)]
#![allow(clippy::uninlined_format_args)]

// need a main anyway, use it get rid of unused warnings too
pub fn main() {
    let x = Some(5);
    // the easiest cases
    let _ = x.and_then(Some);
    //~^ bind_instead_of_map
    let _ = x.and_then(|o| Some(o + 1));
    //~^ bind_instead_of_map
    // and an easy counter-example
    let _ = x.and_then(|o| if o < 32 { Some(o) } else { None });

    // Different type
    let x: Result<u32, &str> = Ok(1);
    let _ = x.and_then(Ok);
    //~^ bind_instead_of_map
}

pub fn foo() -> Option<String> {
    let x = Some(String::from("hello"));
    Some("hello".to_owned()).and_then(|s| Some(format!("{}{}", s, x?)))
}

pub fn example2(x: bool) -> Option<&'static str> {
    Some("a").and_then(|s| Some(if x { s } else { return None }))
}
