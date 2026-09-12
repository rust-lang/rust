fn produces_string() -> Option<String> {
    Some("my cool string".to_owned())
}

fn takes_str_but_too_many_refs(_: &str) -> Option<()> {
    Some(())
}

fn no_args() -> Option<()> {
    Some(())
}

unsafe fn takes_str_but_unsafe(_: String) -> Option<()> {
    Some(())
}

struct TypeWithoutDeref;

fn main() {
    let _ = produces_string().and_then(takes_str_but_too_many_refs);
    //~^ ERROR type mismatch in function arguments
    let _ = produces_string().and_then(takes_str_but_unsafe);
    //~^ ERROR expected an `FnOnce(String)` closure, found `unsafe fn(String) -> Option<()> {takes_str_but_unsafe}`
    let _ = produces_string().and_then(no_args);
    //~^ ERROR function is expected to take 1 argument, but it takes 0 arguments
    let _ = Some(TypeWithoutDeref).and_then(takes_str_but_too_many_refs);
    //~^ ERROR type mismatch in function arguments
}
