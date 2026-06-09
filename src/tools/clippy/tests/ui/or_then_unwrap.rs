#![warn(clippy::or_then_unwrap)]
#![allow(clippy::map_identity, clippy::let_unit_value, clippy::unnecessary_literal_unwrap)]

struct SomeStruct;
impl SomeStruct {
    fn or(self, _: Option<Self>) -> Self {
        self
    }
    fn unwrap(&self) {}
}

struct SomeOtherStruct;
impl SomeOtherStruct {
    fn or(self) -> Self {
        self
    }
    fn unwrap(&self) {}
}

fn main() {
    let option: Option<&str> = None;
    let _ = option.or(Some("fallback")).unwrap(); // should trigger lint
    //
    //~^^ or_then_unwrap

    let result: Result<&str, &str> = Err("Error");
    let _ = result.or::<&str>(Ok("fallback")).unwrap(); // should trigger lint
    //
    //~^^ or_then_unwrap

    // Call with macro should preserve the macro call rather than expand it
    let option: Option<Vec<&str>> = None;
    let _ = option.or(Some(vec!["fallback"])).unwrap(); // should trigger lint
    //
    //~^^ or_then_unwrap

    // as part of a method chain
    let option: Option<&str> = None;
    let _ = option.map(|v| v).or(Some("fallback")).unwrap().to_string().chars(); // should trigger lint
    //
    //~^^ or_then_unwrap

    // Not Option/Result
    let instance = SomeStruct {};
    let _ = instance.or(Some(SomeStruct {})).unwrap(); // should not trigger lint

    // or takes no argument
    let instance = SomeOtherStruct {};
    let _ = instance.or().unwrap(); // should not trigger lint and should not panic

    // None in or
    let option: Option<&str> = None;
    let _ = option.or(None).unwrap(); // should not trigger lint

    // Not Err in or
    let result: Result<&str, &str> = Err("Error");
    let _ = result.or::<&str>(Err("Other Error")).unwrap(); // should not trigger lint

    // other function between
    let option: Option<&str> = None;
    let _ = option.or(Some("fallback")).map(|v| v).unwrap(); // should not trigger lint
}
