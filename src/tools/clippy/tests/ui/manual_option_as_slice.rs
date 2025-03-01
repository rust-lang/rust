#![warn(clippy::manual_option_as_slice)]
#![allow(clippy::redundant_closure, clippy::unwrap_or_default)]

fn check(x: Option<u32>) {
    _ = match x.as_ref() {
        //~^ manual_option_as_slice
        Some(f) => std::slice::from_ref(f),
        None => &[],
    };

    _ = if let Some(f) = x.as_ref() {
        //~^ manual_option_as_slice

        std::slice::from_ref(f)
    } else {
        &[]
    };

    _ = x.as_ref().map_or(&[][..], std::slice::from_ref);
    //~^ manual_option_as_slice

    _ = x.as_ref().map_or_else(Default::default, std::slice::from_ref);
    //~^ manual_option_as_slice

    _ = x.as_ref().map(std::slice::from_ref).unwrap_or_default();
    //~^ manual_option_as_slice

    _ = x.as_ref().map_or_else(|| &[42][..0], std::slice::from_ref);
    //~^ manual_option_as_slice

    {
        use std::slice::from_ref;
        _ = x.as_ref().map_or_else(<&[_]>::default, from_ref);
        //~^ manual_option_as_slice
    }

    // possible false positives
    let y = x.as_ref();
    _ = match y {
        // as_ref outside
        Some(f) => &[f][..],
        None => &[][..],
    };
    _ = match x.as_ref() {
        Some(f) => std::slice::from_ref(f),
        None => &[0],
    };
    _ = match x.as_ref() {
        Some(42) => &[23],
        Some(f) => std::slice::from_ref(f),
        None => &[],
    };
    let b = &[42];
    _ = if let Some(_f) = x.as_ref() {
        std::slice::from_ref(b)
    } else {
        &[]
    };
    _ = x.as_ref().map_or(&[42][..], std::slice::from_ref);
    _ = x.as_ref().map_or_else(|| &[42][..1], std::slice::from_ref);
    _ = x.as_ref().map(|f| std::slice::from_ref(f)).unwrap_or_default();
}

#[clippy::msrv = "1.74"]
fn check_msrv(x: Option<u32>) {
    _ = x.as_ref().map_or(&[][..], std::slice::from_ref);
}

fn main() {
    check(Some(1));
    check_msrv(Some(175));
}
