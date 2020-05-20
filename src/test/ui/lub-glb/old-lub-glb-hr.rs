// Test that we give a note when the old LUB/GLB algorithm would have
// succeeded but the new code (which requires equality) gives an
// error. However, now that we handle subtyping correctly, we no
// longer get an error, because we recognize these two types as
// equivalent!

fn foo(
    x: fn(&u8, &u8),
    y: for<'a> fn(&'a u8, &'a u8),
) {
    // The two types above are actually equivalent. With the older
    // leak check, though, we didn't consider them as equivalent, and
    // hence we gave errors. But now we've fixed that.
    let z = match 22 {
        0 => x,
        _ => y,
    };
}

fn foo_cast(
    x: fn(&u8, &u8),
    y: for<'a> fn(&'a u8, &'a u8),
) {
    let z = match 22 {
        // No error with an explicit cast:
        0 => x as for<'a> fn(&'a u8, &'a u8),
        _ => y,
    };
}

fn bar(
    x: for<'a, 'b> fn(&'a u8, &'b u8)-> &'a u8,
    y: for<'a> fn(&'a u8, &'a u8) -> &'a u8,
) {
    // The two types above are not equivalent. With the older LUB/GLB
    // algorithm, this may have worked (I don't remember), but now it
    // doesn't because we require equality.
    let z = match 22 {
        0 => x,
        _ => y, //~ ERROR `match` arms have incompatible types
    };
}

fn bar_cast(
    x: for<'a, 'b> fn(&'a u8, &'b u8)-> &'a u8,
    y: for<'a> fn(&'a u8, &'a u8) -> &'a u8,
) {
    // But we can *upcast* explicitly the type of `x` and figure
    // things out:
    let z = match 22 {
        0 => x as for<'a> fn(&'a u8, &'a u8) -> &'a u8,
        _ => y,
    };
}

fn main() {
}
