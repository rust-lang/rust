#![warn(clippy::option_if_let_else)]
#![allow(
    clippy::ref_option_ref,
    clippy::equatable_if_let,
    clippy::let_unit_value,
    clippy::redundant_locals,
    clippy::manual_midpoint,
    clippy::manual_unwrap_or_default,
    clippy::manual_unwrap_or
)]

fn bad1(string: Option<&str>) -> (bool, &str) {
    if let Some(x) = string {
        //~^ option_if_let_else
        (true, x)
    } else {
        (false, "hello")
    }
}

fn else_if_option(string: Option<&str>) -> Option<(bool, &str)> {
    if string.is_none() {
        None
    } else if let Some(x) = string {
        Some((true, x))
    } else {
        Some((false, ""))
    }
}

fn unop_bad(string: &Option<&str>, mut num: Option<i32>) {
    let _ = if let Some(s) = *string { s.len() } else { 0 };
    //~^ option_if_let_else
    let _ = if let Some(s) = &num { s } else { &0 };
    //~^ option_if_let_else
    let _ = if let Some(s) = &mut num {
        //~^ option_if_let_else
        *s += 1;
        s
    } else {
        &0
    };
    let _ = if let Some(ref s) = num { s } else { &0 };
    //~^ option_if_let_else
    let _ = if let Some(mut s) = num {
        //~^ option_if_let_else
        s += 1;
        s
    } else {
        0
    };
    let _ = if let Some(ref mut s) = num {
        //~^ option_if_let_else
        *s += 1;
        s
    } else {
        &0
    };
}

fn longer_body(arg: Option<u32>) -> u32 {
    if let Some(x) = arg {
        //~^ option_if_let_else
        let y = x * x;
        y * y
    } else {
        13
    }
}

fn impure_else(arg: Option<i32>) {
    let side_effect = || {
        println!("return 1");
        1
    };
    let _ = if let Some(x) = arg {
        //~^ option_if_let_else
        x
    } else {
        // map_or_else must be suggested
        side_effect()
    };
}

fn test_map_or_else(arg: Option<u32>) {
    let _ = if let Some(x) = arg {
        //~^ option_if_let_else
        x * x * x * x
    } else {
        let mut y = 1;
        y = (y + 2 / y) / 2;
        y = (y + 2 / y) / 2;
        y
    };
}

fn negative_tests(arg: Option<u32>) -> u32 {
    let _ = if let Some(13) = arg { "unlucky" } else { "lucky" };
    for _ in 0..10 {
        let _ = if let Some(x) = arg {
            x
        } else {
            continue;
        };
    }
    let _ = if let Some(x) = arg {
        return x;
    } else {
        5
    };
    7
}

// #7973
fn pattern_to_vec(pattern: &str) -> Vec<String> {
    pattern
        .trim_matches('/')
        .split('/')
        .flat_map(|s| {
            if let Some(idx) = s.find('.') {
                //~^ option_if_let_else
                vec![s[..idx].to_string(), s[idx..].to_string()]
            } else {
                vec![s.to_string()]
            }
        })
        .collect::<Vec<_>>()
}

// #10335
fn test_result_impure_else(variable: Result<u32, &str>) -> bool {
    if let Ok(binding) = variable {
        //~^ option_if_let_else
        println!("Ok {binding}");
        true
    } else {
        println!("Err");
        false
    }
}

enum DummyEnum {
    One(u8),
    Two,
}

// should not warn since there is a complex subpat
// see #7991
fn complex_subpat() -> DummyEnum {
    let x = Some(DummyEnum::One(1));
    let _ = if let Some(_one @ DummyEnum::One(..)) = x { 1 } else { 2 };
    DummyEnum::Two
}

fn main() {
    let optional = Some(5);
    let _ = if let Some(x) = optional { x + 2 } else { 5 };
    //~^ option_if_let_else
    let _ = bad1(None);
    let _ = else_if_option(None);
    unop_bad(&None, None);
    let _ = longer_body(None);
    test_map_or_else(None);
    test_result_impure_else(Ok(42));
    let _ = negative_tests(None);
    let _ = impure_else(None);

    let _ = if let Some(x) = Some(0) {
        //~^ option_if_let_else
        loop {
            if x == 0 {
                break x;
            }
        }
    } else {
        0
    };

    // #7576
    const fn _f(x: Option<u32>) -> u32 {
        // Don't lint, `map_or` isn't const
        if let Some(x) = x { x } else { 10 }
    }

    // #5822
    let s = String::new();
    // Don't lint, `Some` branch consumes `s`, but else branch uses `s`
    let _ = if let Some(x) = Some(0) {
        let s = s;
        s.len() + x
    } else {
        s.len()
    };

    let s = String::new();
    // Lint, both branches immutably borrow `s`.
    let _ = if let Some(x) = Some(0) { s.len() + x } else { s.len() };
    //~^ option_if_let_else

    let s = String::new();
    // Lint, `Some` branch consumes `s`, but else branch doesn't use `s`.
    let _ = if let Some(x) = Some(0) {
        //~^ option_if_let_else
        let s = s;
        s.len() + x
    } else {
        1
    };

    let s = Some(String::new());
    // Don't lint, `Some` branch borrows `s`, but else branch consumes `s`
    let _ = if let Some(x) = &s {
        x.len()
    } else {
        let _s = s;
        10
    };

    let mut s = Some(String::new());
    // Don't lint, `Some` branch mutably borrows `s`, but else branch also borrows  `s`
    let _ = if let Some(x) = &mut s {
        x.push_str("test");
        x.len()
    } else {
        let _s = &s;
        10
    };

    async fn _f1(x: u32) -> u32 {
        x
    }

    async fn _f2() {
        // Don't lint. `await` can't be moved into a closure.
        let _ = if let Some(x) = Some(0) { _f1(x).await } else { 0 };
    }

    let _ = pattern_to_vec("hello world");
    let _ = complex_subpat();

    // issue #8492
    let _ = match s {
        //~^ option_if_let_else
        Some(string) => string.len(),
        None => 1,
    };
    let _ = match Some(10) {
        //~^ option_if_let_else
        Some(a) => a + 1,
        None => 5,
    };

    let res: Result<i32, i32> = Ok(5);
    let _ = match res {
        //~^ option_if_let_else
        Ok(a) => a + 1,
        _ => 1,
    };
    let _ = match res {
        //~^ option_if_let_else
        Err(_) => 1,
        Ok(a) => a + 1,
    };
    let _ = if let Ok(a) = res { a + 1 } else { 5 };
    //~^ option_if_let_else
}

#[allow(dead_code)]
fn issue9742() -> Option<&'static str> {
    // should not lint because of guards
    match Some("foo  ") {
        Some(name) if name.starts_with("foo") => Some(name.trim()),
        _ => None,
    }
}

mod issue10729 {
    #![allow(clippy::unit_arg, dead_code)]

    pub fn reproduce(initial: &Option<String>) {
        // 👇 needs `.as_ref()` because initial is an `&Option<_>`
        let _ = match initial {
            //~^ option_if_let_else
            Some(value) => do_something(value),
            None => 42,
        };
    }

    pub fn reproduce2(initial: &mut Option<String>) {
        let _ = match initial {
            //~^ option_if_let_else
            Some(value) => do_something2(value),
            None => 42,
        };
    }

    fn do_something(_value: &str) -> u32 {
        todo!()
    }
    fn do_something2(_value: &mut str) -> u32 {
        todo!()
    }
}

fn issue11429() {
    use std::collections::HashMap;

    macro_rules! new_map {
        () => {{ HashMap::new() }};
    }

    let opt: Option<HashMap<u8, u8>> = None;

    let mut _hashmap = if let Some(hm) = &opt {
        //~^ option_if_let_else
        hm.clone()
    } else {
        HashMap::new()
    };

    let mut _hm = if let Some(hm) = &opt { hm.clone() } else { new_map!() };
    //~^ option_if_let_else
}

fn issue11893() {
    use std::io::Write;
    let mut output = std::io::stdout().lock();
    if let Some(name) = Some("stuff") {
        writeln!(output, "{name:?}").unwrap();
    } else {
        panic!("Haven't thought about this condition.");
    }
}

mod issue13964 {
    #[derive(Debug)]
    struct A(Option<String>);

    fn foo(a: A) {
        let _ = match a.0 {
            Some(x) => x,
            None => panic!("{a:?} is invalid."),
        };
    }

    fn bar(a: A) {
        let _ = if let Some(x) = a.0 {
            x
        } else {
            panic!("{a:?} is invalid.")
        };
    }
}

mod issue11059 {
    use std::fmt::Debug;

    fn box_coercion_unsize(o: Option<i32>) -> Box<dyn Debug> {
        if let Some(o) = o { Box::new(o) } else { Box::new("foo") }
    }

    static S: String = String::new();

    fn deref_with_overload(o: Option<&str>) -> &str {
        if let Some(o) = o { o } else { &S }
    }
}

fn issue15379() {
    let opt = Some(0usize);
    let opt_raw_ptr = &opt as *const Option<usize>;
    let _ = unsafe { if let Some(o) = *opt_raw_ptr { o } else { 1 } };
    //~^ option_if_let_else
}

fn issue15002() {
    let res: Result<String, ()> = Ok("_".to_string());
    let _ = match res {
        //~^ option_if_let_else
        Ok(s) => s.clone(),
        Err(_) => String::new(),
    };
}
