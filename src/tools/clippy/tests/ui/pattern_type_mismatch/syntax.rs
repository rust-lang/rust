#![allow(clippy::all)]
#![warn(clippy::pattern_type_mismatch)]

fn main() {}

fn syntax_match() {
    let ref_value = &Some(&Some(42));

    // not ok
    match ref_value {
        Some(_) => (),
        None => (),
    }

    // ok
    match ref_value {
        &Some(_) => (),
        &None => (),
    }
    match *ref_value {
        Some(_) => (),
        None => (),
    }
}

fn syntax_if_let() {
    let ref_value = &Some(42);

    // not ok
    if let Some(_) = ref_value {}

    // ok
    if let &Some(_) = ref_value {}
    if let Some(_) = *ref_value {}
}

fn syntax_while_let() {
    let ref_value = &Some(42);

    // not ok
    while let Some(_) = ref_value {
        break;
    }

    // ok
    while let &Some(_) = ref_value {
        break;
    }
    while let Some(_) = *ref_value {
        break;
    }
}

fn syntax_for() {
    let ref_value = &Some(23);
    let slice = &[(2, 3), (4, 2)];

    // not ok
    for (_a, _b) in slice.iter() {}

    // ok
    for &(_a, _b) in slice.iter() {}
}

fn syntax_let() {
    let ref_value = &(2, 3);

    // not ok
    let (_n, _m) = ref_value;

    // ok
    let &(_n, _m) = ref_value;
    let (_n, _m) = *ref_value;
}

fn syntax_fn() {
    // not ok
    fn foo((_a, _b): &(i32, i32)) {}

    // ok
    fn foo_ok_1(&(_a, _b): &(i32, i32)) {}
}

fn syntax_closure() {
    fn foo<F>(f: F)
    where
        F: FnOnce(&(i32, i32)),
    {
    }

    // not ok
    foo(|(_a, _b)| ());

    // ok
    foo(|&(_a, _b)| ());
}

fn macro_with_expression() {
    macro_rules! matching_macro {
        ($e:expr) => {
            $e
        };
    }
    let value = &Some(23);

    // not ok
    matching_macro!(match value {
        Some(_) => (),
        _ => (),
    });

    // ok
    matching_macro!(match value {
        &Some(_) => (),
        _ => (),
    });
    matching_macro!(match *value {
        Some(_) => (),
        _ => (),
    });
}

fn macro_expansion() {
    macro_rules! matching_macro {
        ($e:expr) => {
            // not ok
            match $e {
                Some(_) => (),
                _ => (),
            }

            // ok
            match $e {
                &Some(_) => (),
                _ => (),
            }
            match *$e {
                Some(_) => (),
                _ => (),
            }
        };
    }

    let value = &Some(23);
    matching_macro!(value);
}
