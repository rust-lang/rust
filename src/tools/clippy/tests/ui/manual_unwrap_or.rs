#![allow(dead_code)]
#![allow(
    unused_variables,
    clippy::unnecessary_wraps,
    clippy::unnecessary_literal_unwrap,
    clippy::manual_unwrap_or_default
)]

fn option_unwrap_or() {
    // int case
    match Some(1) {
        //~^ manual_unwrap_or
        Some(i) => i,
        None => 42,
    };

    // int case reversed
    match Some(1) {
        //~^ manual_unwrap_or
        None => 42,
        Some(i) => i,
    };

    // richer none expr
    match Some(1) {
        //~^ manual_unwrap_or
        Some(i) => i,
        None => 1 + 42,
    };

    // multiline case
    #[rustfmt::skip]
    match Some(1) {
    //~^ manual_unwrap_or
        Some(i) => i,
        None => {
            42 + 42
                + 42 + 42 + 42
                + 42 + 42 + 42
        }
    };

    // string case
    match Some("Bob") {
        //~^ manual_unwrap_or
        Some(i) => i,
        None => "Alice",
    };

    // don't lint
    match Some(1) {
        Some(i) => i + 2,
        None => 42,
    };
    match Some(1) {
        Some(i) => i,
        None => return,
    };
    for j in 0..4 {
        match Some(j) {
            Some(i) => i,
            None => continue,
        };
        match Some(j) {
            Some(i) => i,
            None => break,
        };
    }

    // cases where the none arm isn't a constant expression
    // are not linted due to potential ownership issues

    // ownership issue example, don't lint
    struct NonCopyable;
    let mut option: Option<NonCopyable> = None;
    match option {
        Some(x) => x,
        None => {
            option = Some(NonCopyable);
            // some more code ...
            option.unwrap()
        },
    };

    // ownership issue example, don't lint
    let option: Option<&str> = None;
    match option {
        Some(s) => s,
        None => &format!("{} {}!", "hello", "world"),
    };

    if let Some(x) = Some(1) {
        //~^ manual_unwrap_or
        x
    } else {
        42
    };

    //don't lint
    if let Some(x) = Some(1) {
        x + 1
    } else {
        42
    };
    if let Some(x) = Some(1) {
        x
    } else {
        return;
    };
    for j in 0..4 {
        if let Some(x) = Some(j) {
            x
        } else {
            continue;
        };
        if let Some(x) = Some(j) {
            x
        } else {
            break;
        };
    }
}

fn result_unwrap_or() {
    // int case
    match Ok::<i32, &str>(1) {
        //~^ manual_unwrap_or
        Ok(i) => i,
        Err(_) => 42,
    };

    // int case, scrutinee is a binding
    let a = Ok::<i32, &str>(1);
    match a {
        //~^ manual_unwrap_or
        Ok(i) => i,
        Err(_) => 42,
    };

    // int case, suggestion must surround Result expr with parentheses
    match Ok(1) as Result<i32, &str> {
        //~^ manual_unwrap_or
        Ok(i) => i,
        Err(_) => 42,
    };

    // method call case, suggestion must not surround Result expr `s.method()` with parentheses
    struct S;
    impl S {
        fn method(self) -> Option<i32> {
            Some(42)
        }
    }
    let s = S {};
    match s.method() {
        //~^ manual_unwrap_or
        Some(i) => i,
        None => 42,
    };

    // int case reversed
    match Ok::<i32, &str>(1) {
        //~^ manual_unwrap_or
        Err(_) => 42,
        Ok(i) => i,
    };

    // richer none expr
    match Ok::<i32, &str>(1) {
        //~^ manual_unwrap_or
        Ok(i) => i,
        Err(_) => 1 + 42,
    };

    // multiline case
    #[rustfmt::skip]
    match Ok::<i32, &str>(1) {
    //~^ manual_unwrap_or
        Ok(i) => i,
        Err(_) => {
            42 + 42
                + 42 + 42 + 42
                + 42 + 42 + 42
        }
    };

    // string case
    match Ok::<&str, &str>("Bob") {
        //~^ manual_unwrap_or
        Ok(i) => i,
        Err(_) => "Alice",
    };

    // don't lint
    match Ok::<i32, &str>(1) {
        Ok(i) => i + 2,
        Err(_) => 42,
    };
    match Ok::<i32, &str>(1) {
        Ok(i) => i,
        Err(_) => return,
    };
    for j in 0..4 {
        match Ok::<i32, &str>(j) {
            Ok(i) => i,
            Err(_) => continue,
        };
        match Ok::<i32, &str>(j) {
            Ok(i) => i,
            Err(_) => break,
        };
    }

    // don't lint, Err value is used
    match Ok::<&str, &str>("Alice") {
        Ok(s) => s,
        Err(s) => s,
    };
    match Ok::<&str, &str>("Alice") {
        //~^ manual_unwrap_or
        Ok(s) => s,
        Err(s) => "Bob",
    };

    if let Ok(x) = Ok::<i32, i32>(1) {
        //~^ manual_unwrap_or
        x
    } else {
        42
    };

    //don't lint
    if let Ok(x) = Ok::<i32, i32>(1) {
        x + 1
    } else {
        42
    };
    if let Ok(x) = Ok::<i32, i32>(1) {
        x
    } else {
        return;
    };
    for j in 0..4 {
        if let Ok(x) = Ok::<i32, i32>(j) {
            x
        } else {
            continue;
        };
        if let Ok(x) = Ok::<i32, i32>(j) {
            x
        } else {
            break;
        };
    }
}

// don't lint in const fn
const fn const_fn_option_unwrap_or() {
    match Some(1) {
        Some(s) => s,
        None => 0,
    };
}

const fn const_fn_result_unwrap_or() {
    match Ok::<&str, &str>("Alice") {
        Ok(s) => s,
        Err(_) => "Bob",
    };
}

mod issue6965 {
    macro_rules! some_macro {
        () => {
            if 1 > 2 { Some(1) } else { None }
        };
    }

    fn test() {
        let _ = match some_macro!() {
            //~^ manual_unwrap_or
            Some(val) => val,
            None => 0,
        };
    }
}

use std::rc::Rc;
fn format_name(name: Option<&Rc<str>>) -> &str {
    match name {
        None => "<anon>",
        Some(name) => name,
    }
}

fn implicit_deref_ref() {
    let _: &str = match Some(&"bye") {
        None => "hi",
        Some(s) => s,
    };
}

mod issue_13018 {
    use std::collections::HashMap;

    type RefName = i32;
    pub fn get(index: &HashMap<usize, Vec<RefName>>, id: usize) -> &[RefName] {
        if let Some(names) = index.get(&id) { names } else { &[] }
    }

    pub fn get_match(index: &HashMap<usize, Vec<RefName>>, id: usize) -> &[RefName] {
        match index.get(&id) {
            Some(names) => names,
            None => &[],
        }
    }
}

fn implicit_deref(v: Vec<String>) {
    let _ = if let Some(s) = v.first() { s } else { "" };
}

fn allowed_manual_unwrap_or_zero() -> u32 {
    if let Some(x) = Some(42) {
        //~^ manual_unwrap_or
        x
    } else {
        0
    }
}

fn main() {}
