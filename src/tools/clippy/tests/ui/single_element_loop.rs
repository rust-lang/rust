// Tests from for_loop.rs that don't have suggestions

#![allow(clippy::single_range_in_vec_init)]

#[warn(clippy::single_element_loop)]
fn main() {
    let item1 = 2;
    for item in &[item1] {
        //~^ single_element_loop
        dbg!(item);
    }

    for item in [item1].iter() {
        //~^ single_element_loop
        dbg!(item);
    }

    for item in &[0..5] {
        //~^ single_element_loop
        dbg!(item);
    }

    for item in [0..5].iter_mut() {
        //~^ single_element_loop
        dbg!(item);
    }

    for item in [0..5] {
        //~^ single_element_loop
        dbg!(item);
    }

    for item in [0..5].into_iter() {
        //~^ single_element_loop
        dbg!(item);
    }

    // should not lint (issue #10018)
    for e in [42] {
        if e > 0 {
            continue;
        }
    }

    // should not lint (issue #10018)
    for e in [42] {
        if e > 0 {
            break;
        }
    }

    // should lint (issue #10018)
    for _ in [42] {
        //~^ single_element_loop
        let _f = |n: u32| {
            for i in 0..n {
                if i > 10 {
                    dbg!(i);
                    break;
                }
            }
        };
    }

    // Should lint with correct suggestion (issue #12782)
    let res_void: Result<bool, bool> = Ok(true);

    for (Ok(mut _x) | Err(mut _x)) in [res_void] {
        //~^ single_element_loop
        let ptr: *const bool = std::ptr::null();
    }
}
