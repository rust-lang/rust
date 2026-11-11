fn req_fn(_: impl Fn(&'static str) -> String) {}
fn req_fn_mut(_: impl FnMut(&'static str) -> String) {}

fn test_fn_mut_passed_as_mut_ref_to_fn() {
    let mut v = Vec::new();
    let mut accumulate = |x| {
        //~^ ERROR E0525
        v.push(x);
        v.join("/")
    };
    req_fn(&mut accumulate);
}

fn test_fn_once_passed_as_mut_ref_to_fn() {
    let s = String::new();
    let mut consume = move |_x| {
        //~^ ERROR E0525
        drop(s);
        String::new()
    };
    req_fn(&mut consume);
}

fn test_fn_once_passed_as_mut_ref_to_fn_mut() {
    let s = String::new();
    let mut consume = move |_x| {
        //~^ ERROR E0525
        drop(s);
        String::new()
    };
    req_fn_mut(&mut consume);
}

fn test_double_ref_fn_mut_passed_to_fn() {
    let mut v = Vec::new();
    let mut accumulate = |x| {
        //~^ ERROR E0525
        v.push(x);
        v.join("/")
    };
    req_fn(&mut &mut accumulate);
}

fn test_fn_passed_as_mut_ref_to_fn() {
    let mut pure_closure = |x: &'static str| x.to_string();
    req_fn(&mut pure_closure);
    //~^ ERROR E0277
}

fn test_nested_closure_conservative_fallback() {
    let mut v = Vec::new();
    let s = String::new();
    let mut outer = |_x| {
        v.push("a");
        let inner = || drop(s);
        inner();
        v.join("/")
    };
    req_fn(&mut outer);
    //~^ ERROR E0277
}

fn test_unresolved_integer_fallback_copy() {
    let x = 0;
    let mut c = |_x| {
        let _y = x;
        String::new()
    };
    req_fn(&mut c);
    //~^ ERROR E0277
}

fn main() {}
