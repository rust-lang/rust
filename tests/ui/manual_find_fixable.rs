#![warn(clippy::manual_find)]
#![allow(unused)]
#![allow(clippy::needless_return, clippy::uninlined_format_args)]

use std::collections::HashMap;

const ARRAY: &[u32; 5] = &[2, 7, 1, 9, 3];

fn lookup(n: u32) -> Option<u32> {
    for &v in ARRAY {
        //~^ manual_find
        if v == n {
            return Some(v);
        }
    }
    None
}

fn with_pat(arr: Vec<(u32, u32)>) -> Option<u32> {
    for (a, _) in arr {
        //~^ manual_find
        if a % 2 == 0 {
            return Some(a);
        }
    }
    None
}

struct Data {
    name: String,
    is_true: bool,
}
fn with_struct(arr: Vec<Data>) -> Option<Data> {
    for el in arr {
        //~^ manual_find
        if el.name.len() == 10 {
            return Some(el);
        }
    }
    None
}

struct Tuple(usize, usize);
fn with_tuple_struct(arr: Vec<Tuple>) -> Option<usize> {
    for Tuple(a, _) in arr {
        //~^ manual_find
        if a >= 3 {
            return Some(a);
        }
    }
    None
}

struct A;
impl A {
    fn should_keep(&self) -> bool {
        true
    }
}
fn with_method_call(arr: Vec<A>) -> Option<A> {
    for el in arr {
        //~^ manual_find
        if el.should_keep() {
            return Some(el);
        }
    }
    None
}

fn with_closure(arr: Vec<u32>) -> Option<u32> {
    let f = |el: u32| -> u32 { el + 10 };
    for el in arr {
        //~^ manual_find
        if f(el) == 20 {
            return Some(el);
        }
    }
    None
}

fn with_closure2(arr: HashMap<String, i32>) -> Option<i32> {
    let f = |el: i32| -> bool { el == 10 };
    for &el in arr.values() {
        //~^ manual_find
        if f(el) {
            return Some(el);
        }
    }
    None
}

fn with_bool(arr: Vec<Data>) -> Option<Data> {
    for el in arr {
        //~^ manual_find
        if el.is_true {
            return Some(el);
        }
    }
    None
}

fn with_side_effects(arr: Vec<u32>) -> Option<u32> {
    for v in arr {
        if v == 1 {
            println!("side effect");
            return Some(v);
        }
    }
    None
}

fn with_else(arr: Vec<u32>) -> Option<u32> {
    for el in arr {
        if el % 2 == 0 {
            return Some(el);
        } else {
            println!("{}", el);
        }
    }
    None
}

fn tuple_with_ref(v: [(u8, &u8); 3]) -> Option<u8> {
    for (_, &x) in v {
        //~^ manual_find
        if x > 10 {
            return Some(x);
        }
    }
    None
}

fn ref_to_tuple_with_ref(v: &[(u8, &u8)]) -> Option<u8> {
    for &(_, &x) in v {
        //~^ manual_find
        if x > 10 {
            return Some(x);
        }
    }
    None
}

fn explicit_ret(arr: Vec<i32>) -> Option<i32> {
    for x in arr {
        //~^ manual_find
        if x >= 5 {
            return Some(x);
        }
    }
    return None;
}

fn plus_one(a: i32) -> Option<i32> {
    Some(a + 1)
}
fn fn_instead_of_some(a: &[i32]) -> Option<i32> {
    for &x in a {
        if x == 1 {
            return plus_one(x);
        }
    }
    None
}

fn for_in_condition(a: &[i32], b: bool) -> Option<i32> {
    if b {
        for &x in a {
            if x == 1 {
                return Some(x);
            }
        }
    }
    None
}

fn intermediate_statements(a: &[i32]) -> Option<i32> {
    for &x in a {
        if x == 1 {
            return Some(x);
        }
    }

    println!("side effect");

    None
}

fn mixed_binding_modes(arr: Vec<(i32, String)>) -> Option<i32> {
    for (x, mut s) in arr {
        if x == 1 && s.as_mut_str().len() == 2 {
            return Some(x);
        }
    }
    None
}

fn as_closure() {
    #[rustfmt::skip]
    let f = |arr: Vec<i32>| -> Option<i32> {
        for x in arr {
        //~^ manual_find
            if x < 1 {
                return Some(x);
            }
        }
        None
    };
}

fn in_block(a: &[i32]) -> Option<i32> {
    let should_be_none = {
        for &x in a {
            if x == 1 {
                return Some(x);
            }
        }
        None
    };

    assert!(should_be_none.is_none());

    should_be_none
}

// Not handled yet
fn mut_binding(v: Vec<String>) -> Option<String> {
    for mut s in v {
        if s.as_mut_str().len() > 1 {
            return Some(s);
        }
    }
    None
}

fn subpattern(v: Vec<[u32; 32]>) -> Option<[u32; 32]> {
    for a @ [first, ..] in v {
        if a[12] == first {
            return Some(a);
        }
    }
    None
}

fn two_bindings(v: Vec<(u8, u8)>) -> Option<u8> {
    for (a, n) in v {
        if a == n {
            return Some(a);
        }
    }
    None
}

fn main() {}

mod issue14826 {
    fn adjust_fixable(needle: &str) -> Option<&'static str> {
        for candidate in &["foo", "bar"] {
            //~^ manual_find
            if candidate.eq_ignore_ascii_case(needle) {
                return Some(candidate);
            }
        }
        None
    }

    fn adjust_unfixable(needle: &str) -> Option<*const str> {
        for &candidate in &["foo", "bar"] {
            //~^ manual_find
            if candidate.eq_ignore_ascii_case(needle) {
                return Some(candidate);
            }
        }
        None
    }
}
