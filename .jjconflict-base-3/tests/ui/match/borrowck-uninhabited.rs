// See: rust-lang/rust#146590

enum Never {}

// baseline
fn both_inhabited(x: &mut Result<String, String>) {
    match x {
        &mut Ok(ref mut y) => match x {
        //~^ ERROR: cannot use `*x` because it was mutably borrowed
            &mut Err(ref mut z) => {
                let _y = y;
                let _z = z;
            }
            _ => {}
        },
        _ => {}
    };
}

// this used to be accepted, even though it shouldn't
fn ref_uninhabited(x: &mut Result<Never, String>) {
    match x {
        &mut Ok(ref mut y) => match x {
        //~^ ERROR: cannot use `*x` because it was mutably borrowed
            &mut Err(ref mut z) => {
                let _y = y;
                let _z = z;
            }
            _ => {}
        },
        _ => {}
    };
}

enum Single {
    V(String, String),
}

// arguably this should be rejected as well, but currently it is still accepted
fn single_variant(x: &mut Single) {
    match x {
        &mut Single::V(ref mut y, _) => {
            match x {
                &mut Single::V(_, ref mut z) => {
                    let _y = y;
                    let _z = z;
                }
                _ => {}
            }
        },
        _ => {}
    };
}

fn main() {}
