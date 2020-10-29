// run-rustfix

#![warn(clippy::match_like_matches_macro)]
#![allow(unreachable_patterns, dead_code)]

fn main() {
    let x = Some(5);

    // Lint
    let _y = match x {
        Some(0) => true,
        _ => false,
    };

    // Lint
    let _w = match x {
        Some(_) => true,
        _ => false,
    };

    // Turn into is_none
    let _z = match x {
        Some(_) => false,
        None => true,
    };

    // Lint
    let _zz = match x {
        Some(r) if r == 0 => false,
        _ => true,
    };

    // Lint
    let _zzz = if let Some(5) = x { true } else { false };

    // No lint
    let _a = match x {
        Some(_) => false,
        _ => false,
    };

    // No lint
    let _ab = match x {
        Some(0) => false,
        _ => true,
        None => false,
    };

    enum E {
        A(u32),
        B(i32),
        C,
        D,
    };
    let x = E::A(2);
    {
        // lint
        let _ans = match x {
            E::A(_) => true,
            E::B(_) => true,
            _ => false,
        };
    }
    {
        // lint
        let _ans = match x {
            E::B(_) => false,
            E::C => false,
            _ => true,
        };
    }
    {
        // no lint
        let _ans = match x {
            E::A(_) => false,
            E::B(_) => false,
            E::C => true,
            _ => true,
        };
    }
    {
        // no lint
        let _ans = match x {
            E::A(_) => true,
            E::B(_) => false,
            E::C => false,
            _ => true,
        };
    }
    {
        // no lint
        let _ans = match x {
            E::A(a) if a < 10 => false,
            E::B(a) if a < 10 => false,
            _ => true,
        };
    }
    {
        // no lint
        let _ans = match x {
            E::A(_) => false,
            E::B(a) if a < 10 => false,
            _ => true,
        };
    }
    {
        // no lint
        let _ans = match x {
            E::A(a) => a == 10,
            E::B(_) => false,
            _ => true,
        };
    }
    {
        // no lint
        let _ans = match x {
            E::A(_) => false,
            E::B(_) => true,
            _ => false,
        };
    }
}
