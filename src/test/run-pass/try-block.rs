#![allow(non_camel_case_types)]
#![allow(dead_code)]
// compile-flags: --edition 2018

#![feature(try_blocks)]

struct catch {}

pub fn main() {
    let catch_result: Option<_> = try {
        let x = 5;
        x
    };
    assert_eq!(catch_result, Some(5));

    let mut catch = true;
    while catch { catch = false; }
    assert_eq!(catch, false);

    catch = if catch { false } else { true };
    assert_eq!(catch, true);

    match catch {
        _ => {}
    };

    let catch_err: Result<_, i32> = try {
        Err(22)?;
        1
    };
    assert_eq!(catch_err, Err(22));

    let catch_okay: Result<i32, i32> = try {
        if false { Err(25)?; }
        Ok::<(), i32>(())?;
        28
    };
    assert_eq!(catch_okay, Ok(28));

    let catch_from_loop: Result<i32, i32> = try {
        for i in 0..10 {
            if i < 5 { Ok::<i32, i32>(i)?; } else { Err(i)?; }
        }
        22
    };
    assert_eq!(catch_from_loop, Err(5));

    let cfg_init;
    let _res: Result<(), ()> = try {
        cfg_init = 5;
    };
    assert_eq!(cfg_init, 5);

    let cfg_init_2;
    let _res: Result<(), ()> = try {
        cfg_init_2 = 6;
        Err(())?;
    };
    assert_eq!(cfg_init_2, 6);

    let my_string = "test".to_string();
    let res: Result<&str, ()> = try {
        // Unfortunately, deref doesn't fire here (#49356)
        &my_string[..]
    };
    assert_eq!(res, Ok("test"));

    let my_opt: Option<_> = try { () };
    assert_eq!(my_opt, Some(()));

    let my_opt: Option<_> = try { };
    assert_eq!(my_opt, Some(()));
}
