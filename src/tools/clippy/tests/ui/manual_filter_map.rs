#![allow(dead_code)]
#![warn(clippy::manual_filter_map)]
#![allow(clippy::redundant_closure)] // FIXME suggestion may have redundant closure
#![allow(clippy::useless_vec)]
#![allow(clippy::struct_field_names)]

fn main() {
    // is_some(), unwrap()
    let _ = (0..).filter(|n| to_opt(*n).is_some()).map(|a| to_opt(a).unwrap());
    //~^ manual_filter_map

    // ref pattern, expect()
    let _ = (0..).filter(|&n| to_opt(n).is_some()).map(|a| to_opt(a).expect("hi"));
    //~^ manual_filter_map

    // is_ok(), unwrap_or()
    let _ = (0..).filter(|&n| to_res(n).is_ok()).map(|a| to_res(a).unwrap_or(1));
    //~^ manual_filter_map

    let _ = (1..5)
        .filter(|&x| to_ref(to_opt(x)).is_some())
        //~^ manual_filter_map
        .map(|y| to_ref(to_opt(y)).unwrap());
    let _ = (1..5)
        .filter(|x| to_ref(to_opt(*x)).is_some())
        //~^ manual_filter_map
        .map(|y| to_ref(to_opt(y)).unwrap());

    let _ = (1..5)
        .filter(|&x| to_ref(to_res(x)).is_ok())
        //~^ manual_filter_map
        .map(|y| to_ref(to_res(y)).unwrap());
    let _ = (1..5)
        .filter(|x| to_ref(to_res(*x)).is_ok())
        //~^ manual_filter_map
        .map(|y| to_ref(to_res(y)).unwrap());
}

#[rustfmt::skip]
fn simple_equal() {
    iter::<Option<&u8>>().find(|x| x.is_some()).map(|x| x.cloned().unwrap());
    //~^ manual_find_map
    iter::<&Option<&u8>>().find(|x| x.is_some()).map(|x| x.cloned().unwrap());
    //~^ manual_find_map
    iter::<&Option<String>>().find(|x| x.is_some()).map(|x| x.as_deref().unwrap());
    //~^ manual_find_map
    iter::<Option<&String>>().find(|&x| to_ref(x).is_some()).map(|y| to_ref(y).cloned().unwrap());
    //~^ manual_find_map

    iter::<Result<u8, ()>>().find(|x| x.is_ok()).map(|x| x.unwrap());
    //~^ manual_find_map
    iter::<&Result<u8, ()>>().find(|x| x.is_ok()).map(|x| x.unwrap());
    //~^ manual_find_map
    iter::<&&Result<u8, ()>>().find(|x| x.is_ok()).map(|x| x.unwrap());
    //~^ manual_find_map
    iter::<Result<&u8, ()>>().find(|x| x.is_ok()).map(|x| x.cloned().unwrap());
    //~^ manual_find_map
    iter::<&Result<&u8, ()>>().find(|x| x.is_ok()).map(|x| x.cloned().unwrap());
    //~^ manual_find_map
    iter::<&Result<String, ()>>().find(|x| x.is_ok()).map(|x| x.as_deref().unwrap());
    //~^ manual_find_map
    iter::<Result<&String, ()>>().find(|&x| to_ref(x).is_ok()).map(|y| to_ref(y).cloned().unwrap());
    //~^ manual_find_map
}

fn no_lint() {
    // no shared code
    let _ = (0..).filter(|n| *n > 1).map(|n| n + 1);

    // very close but different since filter() provides a reference
    let _ = (0..).filter(|n| to_opt(n).is_some()).map(|a| to_opt(a).unwrap());

    // similar but different
    let _ = (0..).filter(|n| to_opt(n).is_some()).map(|n| to_res(n).unwrap());
    let _ = (0..)
        .filter(|n| to_opt(n).map(|n| n + 1).is_some())
        .map(|a| to_opt(a).unwrap());
}

fn iter<T>() -> impl Iterator<Item = T> {
    std::iter::empty()
}

fn to_opt<T>(_: T) -> Option<T> {
    unimplemented!()
}

fn to_res<T>(_: T) -> Result<T, ()> {
    unimplemented!()
}

fn to_ref<'a, T>(_: T) -> &'a T {
    unimplemented!()
}

struct Issue8920<'a> {
    option_field: Option<String>,
    result_field: Result<String, ()>,
    ref_field: Option<&'a usize>,
}

fn issue_8920() {
    let mut vec = vec![Issue8920 {
        option_field: Some(String::from("str")),
        result_field: Ok(String::from("str")),
        ref_field: Some(&1),
    }];

    let _ = vec
        .iter()
        .filter(|f| f.option_field.is_some())
        //~^ manual_filter_map
        .map(|f| f.option_field.clone().unwrap());

    let _ = vec
        .iter()
        .filter(|f| f.ref_field.is_some())
        //~^ manual_filter_map
        .map(|f| f.ref_field.cloned().unwrap());

    let _ = vec
        .iter()
        .filter(|f| f.ref_field.is_some())
        //~^ manual_filter_map
        .map(|f| f.ref_field.copied().unwrap());

    let _ = vec
        .iter()
        .filter(|f| f.result_field.is_ok())
        //~^ manual_filter_map
        .map(|f| f.result_field.clone().unwrap());

    let _ = vec
        .iter()
        .filter(|f| f.result_field.is_ok())
        //~^ manual_filter_map
        .map(|f| f.result_field.as_ref().unwrap());

    let _ = vec
        .iter()
        .filter(|f| f.result_field.is_ok())
        //~^ manual_filter_map
        .map(|f| f.result_field.as_deref().unwrap());

    let _ = vec
        .iter_mut()
        .filter(|f| f.result_field.is_ok())
        //~^ manual_filter_map
        .map(|f| f.result_field.as_mut().unwrap());

    let _ = vec
        .iter_mut()
        .filter(|f| f.result_field.is_ok())
        //~^ manual_filter_map
        .map(|f| f.result_field.as_deref_mut().unwrap());

    let _ = vec
        .iter()
        .filter(|f| f.result_field.is_ok())
        //~^ manual_filter_map
        .map(|f| f.result_field.to_owned().unwrap());
}

fn issue8010() {
    #[derive(Clone)]
    enum Enum {
        A(i32),
        B,
    }

    let iter = [Enum::A(123), Enum::B].into_iter();

    let _x = iter.clone().filter(|x| matches!(x, Enum::A(_))).map(|x| match x {
        //~^ manual_filter_map
        Enum::A(s) => s,
        _ => unreachable!(),
    });
    let _x = iter.clone().filter(|x| matches!(x, Enum::B)).map(|x| match x {
        Enum::A(s) => s,
        _ => unreachable!(),
    });
    let _x = iter
        .clone()
        .filter(|x| matches!(x, Enum::A(_)))
        //~^ manual_filter_map
        .map(|x| if let Enum::A(s) = x { s } else { unreachable!() });
    #[allow(clippy::unused_unit)]
    let _x = iter
        .clone()
        .filter(|x| matches!(x, Enum::B))
        .map(|x| if let Enum::B = x { () } else { unreachable!() });
}
