// this isn't auto-fixable now because we produce two similar suggestions

fn produces_string() -> Option<String> {
    Some("my cool string".to_owned())
}

fn takes_str(_: &str) -> Option<()> {
    Some(())
}

fn takes_str_mut(_: &mut str) -> Option<()> {
    Some(())
}

fn generic<T>(_: T) -> Option<()> {
    Some(())
}

fn generic_ref<T>(_: &T) -> Option<()> {
    //~^ HELP consider adjusting the signature so it does not borrow its argument
    Some(())
}

fn main() {
    let _: Option<()> = produces_string().and_then(takes_str);
    //~^ ERROR type mismatch in function arguments
    //~| HELP call `Option::as_deref()` first
    //~| HELP consider wrapping the function in a closure
    let _: Option<Option<()>> = produces_string().map(takes_str);
    //~^ ERROR type mismatch in function arguments
    //~| HELP call `Option::as_deref()` first
    //~| HELP consider wrapping the function in a closure
    let _: Option<Option<()>> = produces_string().map(takes_str_mut);
    //~^ ERROR type mismatch in function arguments
    //~| HELP call `Option::as_deref_mut()` first
    //~| HELP consider wrapping the function in a closure
    let _ = produces_string().and_then(generic);

    let _ = produces_string().and_then(generic_ref);
    //~^ ERROR type mismatch in function arguments
    //~| HELP call `Option::as_deref()` first
    //~| HELP consider wrapping the function in a closure
}
