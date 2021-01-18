#![warn(clippy::if_same_then_else)]
#![allow(
    clippy::blacklisted_name,
    clippy::collapsible_else_if,
    clippy::collapsible_if,
    clippy::ifs_same_cond,
    clippy::needless_return,
    clippy::single_element_loop
)]

fn if_same_then_else2() -> Result<&'static str, ()> {
    if true {
        for _ in &[42] {
            let foo: &Option<_> = &Some::<u8>(42);
            if true {
                break;
            } else {
                continue;
            }
        }
    } else {
        //~ ERROR same body as `if` block
        for _ in &[42] {
            let foo: &Option<_> = &Some::<u8>(42);
            if true {
                break;
            } else {
                continue;
            }
        }
    }

    if true {
        if let Some(a) = Some(42) {}
    } else {
        //~ ERROR same body as `if` block
        if let Some(a) = Some(42) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    } else {
        //~ ERROR same body as `if` block
        if let (1, .., 3) = (1, 2, 3) {}
    }

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
    let _ = if true {
        f32::NAN
    } else {
        //~ ERROR same body as `if` block
        f32::NAN
    };

    if true {
        Ok("foo")?;
    } else {
        //~ ERROR same body as `if` block
        Ok("foo")?;
    }

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
}

fn main() {}
