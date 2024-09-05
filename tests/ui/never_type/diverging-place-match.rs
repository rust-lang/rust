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

fn main() {}
