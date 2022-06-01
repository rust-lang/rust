// run-rustfix
#![allow(dead_code)]
#![warn(clippy::manual_find_map)]
#![allow(clippy::redundant_closure)] // FIXME suggestion may have redundant closure

fn main() {
    // is_some(), unwrap()
    let _ = (0..).find(|n| to_opt(*n).is_some()).map(|a| to_opt(a).unwrap());

    // ref pattern, expect()
    let _ = (0..).find(|&n| to_opt(n).is_some()).map(|a| to_opt(a).expect("hi"));

    // is_ok(), unwrap_or()
    let _ = (0..).find(|&n| to_res(n).is_ok()).map(|a| to_res(a).unwrap_or(1));
}

fn no_lint() {
    // no shared code
    let _ = (0..).filter(|n| *n > 1).map(|n| n + 1);

    // very close but different since filter() provides a reference
    let _ = (0..).find(|n| to_opt(n).is_some()).map(|a| to_opt(a).unwrap());

    // similar but different
    let _ = (0..).find(|n| to_opt(n).is_some()).map(|n| to_res(n).unwrap());
    let _ = (0..)
        .find(|n| to_opt(n).map(|n| n + 1).is_some())
        .map(|a| to_opt(a).unwrap());
}

fn to_opt<T>(_: T) -> Option<T> {
    unimplemented!()
}

fn to_res<T>(_: T) -> Result<T, ()> {
    unimplemented!()
}

struct OptionFoo {
    field: Option<String>,
}

struct ResultFoo {
    field: Result<String, ()>,
}

fn issue_8920() {
    let vec = vec![OptionFoo {
        field: Some(String::from("str")),
    }];
    let _ = vec.iter().find(|f| f.field.is_some()).map(|f| f.field.clone().unwrap());

    let vec = vec![ResultFoo {
        field: Ok(String::from("str")),
    }];
    let _ = vec.iter().find(|f| f.field.is_ok()).map(|f| f.field.clone().unwrap());
}
