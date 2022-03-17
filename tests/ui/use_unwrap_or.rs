#![warn(clippy::use_unwrap_or)]

struct SomeStruct {}
impl SomeStruct {
    fn or(self, _: Option<Self>) -> Self { self }
    fn unwrap(&self){}
}

fn main() {
    let option: Option<&str> = None;
    let _ = option.or(Some("fallback")).unwrap(); // should trigger lint

    let result: Result<&str, &str> = Err("Error");
    let _ = result.or::<&str>(Ok("fallback")).unwrap(); // should trigger lint

    // Not Option/Result
    let instance = SomeStruct {};
    let _ = instance.or(Some(SomeStruct {})).unwrap(); // should not trigger lint

    // None in or
    let option: Option<&str> = None;
    let _ = option.or(None).unwrap(); // should not trigger lint

    // Not Err in or
    let result: Result<&str, &str> = Err("Error");
    let _ = result.or::<&str>(Err("Other Error")).unwrap(); // should not trigger lint
}
