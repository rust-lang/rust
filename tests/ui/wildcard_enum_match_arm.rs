#![warn(clippy::wildcard_enum_match_arm)]

#[derive(Debug)]
enum Maybe<T> {
    Some(T),
    Probably(T),
    None,
}

fn is_it_wildcard<T>(m: Maybe<T>) -> &'static str {
    match m {
        Maybe::Some(_) => "Some",
        _ => "Could be",
    }
}

fn is_it_bound<T>(m: Maybe<T>) -> &'static str {
    match m {
        Maybe::None => "None",
        _other => "Could be",
    }
}

fn is_it_binding(m: Maybe<u32>) -> String {
    match m {
        Maybe::Some(v) => "Large".to_string(),
        n => format!("{:?}", n),
    }
}

fn is_it_binding_exhaustive(m: Maybe<u32>) -> String {
    match m {
        Maybe::Some(v) => "Large".to_string(),
        n @ Maybe::Probably(_) | n @ Maybe::None => format!("{:?}", n),
    }
}

fn is_it_with_guard(m: Maybe<u32>) -> &'static str {
    match m {
        Maybe::Some(v) if v > 100 => "Large",
        _ => "Who knows",
    }
}

fn is_it_exhaustive<T>(m: Maybe<T>) -> &'static str {
    match m {
        Maybe::None => "None",
        Maybe::Some(_) | Maybe::Probably(..) => "Could be",
    }
}

fn is_one_or_three(i: i32) -> bool {
    match i {
        1 | 3 => true,
        _ => false,
    }
}

fn main() {
    println!("{}", is_it_wildcard(Maybe::Some("foo")));

    println!("{}", is_it_bound(Maybe::Some("foo")));

    println!("{}", is_it_binding(Maybe::Some(1)));

    println!("{}", is_it_binding_exhaustive(Maybe::Some(1)));

    println!("{}", is_it_with_guard(Maybe::Some(1)));

    println!("{}", is_it_exhaustive(Maybe::Some("foo")));

    println!("{}", is_one_or_three(2));
}
