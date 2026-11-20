#![warn(clippy::needless_bool)]
#![allow(dead_code, unused, clippy::needless_return)]

// The motivating case from uutils/coreutils#12689: a guard returning a wrapped bool
// followed by a trailing wrapped bool literal.
fn is_sparse(blocks: u64, size: u64) -> Result<bool, ()> {
    if blocks < size / 512 {
        return Ok(true);
    }
    Ok(false)
    //~^^^^ needless_bool
}

fn bare(x: bool) -> bool {
    if x {
        return true;
    }
    false
    //~^^^^ needless_bool
}

fn bare_negated(x: bool) -> bool {
    if x {
        return false;
    }
    true
    //~^^^^ needless_bool
}

fn option_wrapped(x: bool) -> Option<bool> {
    if x {
        return Some(false);
    }
    Some(true)
    //~^^^^ needless_bool
}

fn complex_condition(a: i32, b: i32) -> Result<bool, ()> {
    if a < b && b > 0 {
        return Ok(false);
    }
    Ok(true)
    //~^^^^ needless_bool
}

fn multi_if_complex_previous_if(x: bool, y: bool) -> Result<bool, ()> {
    if y {
        let z = x && y;
        if z {
            return Ok(x);
        }
        return Ok(true);
    }
    if x {
        return Ok(true);
    }
    Ok(false)
    //~^^^^ needless_bool
}

// Do NOT lint: the two values are equal, and the condition might have side effects.
fn same_value(x: bool) -> Result<bool, ()> {
    if x {
        return Ok(true);
    }
    Ok(true)
}

// Do NOT lint: mismatched wrappers.
fn mismatched_wrappers(x: bool) -> Result<bool, ()> {
    if x {
        return Err(());
    }
    Ok(false)
}

// Do NOT lint: the guard body has a side effect besides the return.
fn side_effect(x: bool) -> Result<bool, ()> {
    if x {
        println!("hi");
        return Ok(true);
    }
    Ok(false)
}

// Do NOT lint: not a constructor, just a function call (could have side effects).
fn make(b: bool) -> bool {
    b
}
fn not_a_ctor(x: bool) -> bool {
    if x {
        return make(true);
    }
    make(false)
}

// Do NOT lint: multiple if statements chained one after another.
fn multi_if(x: bool, y: bool) -> Result<bool, ()> {
    if y {
        return Ok(true);
    }
    if x {
        return Ok(true);
    }
    Ok(false)
}

// Do NOT lint: multiple if statements chained one after another, even if they have different return
// values.
fn multi_if_different_values(x: bool, y: bool) -> bool {
    if y {
        return false;
    }
    if x {
        return true;
    }
    false
}

fn main() {}
