#![warn(clippy::if_same_then_else)]
#![allow(
    clippy::disallowed_names,
    clippy::collapsible_else_if,
    clippy::equatable_if_let,
    clippy::collapsible_if,
    clippy::ifs_same_cond,
    clippy::needless_if,
    clippy::needless_return,
    clippy::single_element_loop,
    clippy::branches_sharing_code
)]

fn if_same_then_else2() -> Result<&'static str, ()> {
    if true {
        for _ in &[42] {
            let foo: &Option<_> = &Some::<u8>(42);
            if foo.is_some() {
                break;
            } else {
                continue;
            }
        }
    } else {
        for _ in &[42] {
            let bar: &Option<_> = &Some::<u8>(42);
            if bar.is_some() {
                break;
            } else {
                continue;
            }
        }
    }
    //~^^^^^^^^^^^^^^^^^^^ ERROR: this `if` has identical blocks

    if true {
        if let Some(a) = Some(42) {}
    } else {
        if let Some(a) = Some(42) {}
    }
    //~^^^^^ ERROR: this `if` has identical blocks

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    } else {
        if let (1, .., 3) = (1, 2, 3) {}
    }
    //~^^^^^ ERROR: this `if` has identical blocks

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    } else {
        if let (.., 3) = (1, 2, 3) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    } else {
        if let (.., 4) = (1, 2, 3) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    } else {
        if let (.., 1, 3) = (1, 2, 3) {}
    }

    if true {
        if let Some(42) = None {}
    } else {
        if let Option::Some(42) = None {}
    }

    if true {
        if let Some(42) = None::<u8> {}
    } else {
        if let Some(42) = None {}
    }

    if true {
        if let Some(42) = None::<u8> {}
    } else {
        if let Some(42) = None::<u32> {}
    }

    if true {
        if let Some(a) = Some(42) {}
    } else {
        if let Some(a) = Some(43) {}
    }

    // Same NaNs
    let _ = if true { f32::NAN } else { f32::NAN };
    //~^ ERROR: this `if` has identical blocks

    if true {
        Ok("foo")?;
    } else {
        Ok("foo")?;
    }
    //~^^^^^ ERROR: this `if` has identical blocks

    if true {
        let foo = "";
        return Ok(&foo[0..]);
    } else if false {
        let foo = "bar";
        return Ok(&foo[0..]);
    } else {
        let foo = "";
        return Ok(&foo[0..]);
    }

    if true {
        let foo = "";
        return Ok(&foo[0..]);
    } else if false {
        let foo = "bar";
        return Ok(&foo[0..]);
    } else if true {
        let foo = "";
        return Ok(&foo[0..]);
    } else {
        let foo = "";
        return Ok(&foo[0..]);
    }
    //~^^^^^^^ ERROR: this `if` has identical blocks

    // False positive `if_same_then_else`: `let (x, y)` vs. `let (y, x)`; see issue #3559.
    if true {
        let foo = "";
        let (x, y) = (1, 2);
        return Ok(&foo[x..y]);
    } else {
        let foo = "";
        let (y, x) = (1, 2);
        return Ok(&foo[x..y]);
    }

    // Issue #7579
    let _ = if let Some(0) = None { 0 } else { 0 };

    if true {
        return Err(());
    } else if let Some(0) = None {
        return Err(());
    }

    let _ = if let Some(0) = None {
        0
    } else if let Some(1) = None {
        0
    } else {
        0
    };
}

fn main() {}
