// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(catch_expr)]

struct catch {}

pub fn main() {
    let catch_result: Option<_> = do catch {
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

    let catch_err: Result<_, i32> = do catch {
        Err(22)?;
        1
    };
    assert_eq!(catch_err, Err(22));

    let catch_okay: Result<i32, i32> = do catch {
        if false { Err(25)?; }
        Ok::<(), i32>(())?;
        28
    };
    assert_eq!(catch_okay, Ok(28));

    let catch_from_loop: Result<i32, i32> = do catch {
        for i in 0..10 {
            if i < 5 { Ok::<i32, i32>(i)?; } else { Err(i)?; }
        }
        22
    };
    assert_eq!(catch_from_loop, Err(5));

    let cfg_init;
    let _res: Result<(), ()> = do catch {
        cfg_init = 5;
    };
    assert_eq!(cfg_init, 5);

    let cfg_init_2;
    let _res: Result<(), ()> = do catch {
        cfg_init_2 = 6;
        Err(())?;
    };
    assert_eq!(cfg_init_2, 6);

    let my_string = "test".to_string();
    let res: Result<&str, ()> = do catch {
        // Unfortunately, deref doesn't fire here (#49356)
        &my_string[..]
    };
    assert_eq!(res, Ok("test"));

    let my_opt: Option<_> = do catch { () };
    assert_eq!(my_opt, Some(()));

    let my_opt: Option<_> = do catch { };
    assert_eq!(my_opt, Some(()));
}
