#![feature(never_type)]

fn not_a_read() -> ! {
    unsafe {
    //~^ ERROR mismatched types
        let x: *const ! = 0 as _;
        let _: ! = *x;
        // Since `*x` "diverges" in HIR, but doesn't count as a read in MIR, this
        // is unsound since we act as if it diverges but it doesn't.
    }
}

fn not_a_read_implicit() -> ! {
    unsafe {
    //~^ ERROR mismatched types
        let x: *const ! = 0 as _;
        let _ = *x;
    }
}

fn not_a_read_guide_coercion() -> ! {
    unsafe {
        //~^ ERROR mismatched types
        let x: *const ! = 0 as _;
        let _: () = *x;
        //~^ ERROR mismatched types
    }
}

fn empty_match() -> ! {
    unsafe {
    //~^ ERROR mismatched types
        let x: *const ! = 0 as _;
        match *x { _ => {} };
    }
}

fn field_projection() -> ! {
    unsafe {
    //~^ ERROR mismatched types
        let x: *const (!, ()) = 0 as _;
        let _ = (*x).0;
        // ^ I think this is still UB, but because of the inbounds projection.
    }
}

fn covered_arm() -> ! {
    unsafe {
    //~^ ERROR mismatched types
        let x: *const ! = 0 as _;
        let (_ | 1i32) = *x;
        //~^ ERROR mismatched types
    }
}

// FIXME: This *could* be considered a read of `!`, but we're not that sophisticated..
fn uncovered_arm() -> ! {
    unsafe {
    //~^ ERROR mismatched types
        let x: *const ! = 0 as _;
        let (1i32 | _) = *x;
        //~^ ERROR mismatched types
    }
}

fn coerce_ref_binding() -> ! {
    unsafe {
        let x: *const ! = 0 as _;
        let ref _x: () = *x;
        //~^ ERROR mismatched types
    }
}

fn main() {}
