// run-rustfix
// Tests from for_loop.rs that don't have suggestions

#[warn(clippy::single_element_loop)]
fn main() {
    let item1 = 2;
    for item in &[item1] {
        dbg!(item);
    }

    for item in [item1].iter() {
        dbg!(item);
    }

    for item in &[0..5] {
        dbg!(item);
    }

    for item in [0..5].iter_mut() {
        dbg!(item);
    }

    for item in [0..5] {
        dbg!(item);
    }

    for item in [0..5].into_iter() {
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
        let _f = |n: u32| {
            for i in 0..n {
                if i > 10 {
                    dbg!(i);
                    break;
                }
            }
        };
    }
}
