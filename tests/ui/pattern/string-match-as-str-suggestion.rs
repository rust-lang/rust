//@ run-rustfix

fn main() {
    let s = "yes".to_owned();

    let _ = match s {
        "yes" => Some(true),
        //~^ ERROR mismatched types
        "no" => Some(false),
        //~^ ERROR mismatched types
        _ => None,
    };

    let s2 = String::from("hello");
    if let "hello" = s2 {
        //~^ ERROR mismatched types
    }
}
