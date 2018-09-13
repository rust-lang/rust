// Test that a (partially) mutably borrowed place can be matched on, so long as
// we don't have to read any values that are mutably borrowed to determine
// which arm to take.
//
// Test that we don't allow mutating the value being matched on in a way that
// changes which patterns it matches, until we have chosen an arm.

// compile-flags: -Zdisable-ast-check-for-mutation-in-guard

#![feature(nll)]

fn ok_mutation_in_guard(mut q: i32) {
    match q {
        // OK, mutation doesn't change which patterns g matches
        _ if { q = 1; false } => (),
        _ => (),
    }
}

fn ok_indirect_mutation_in_guard(mut p: &bool) {
    match *p {
        // OK, mutation doesn't change which patterns s matches
        _ if {
            p = &true;
            false
        } => (),
        _ => (),
    }
}

fn mutation_invalidates_pattern_in_guard(mut q: bool) {
    match q {
        // s doesn't match the pattern with the guard by the end of the guard.
        false if {
            q = true; //~ ERROR
            true
        } => (),
        _ => (),
    }
}

fn mutation_invalidates_previous_pattern_in_guard(mut r: bool) {
    match r {
        // s matches a previous pattern by the end of the guard.
        true => (),
        _ if {
            r = true; //~ ERROR
            true
        } => (),
        _ => (),
    }
}

fn match_on_borrowed_early_end(mut s: bool) {
    let h = &mut s;
    match s { //~ ERROR
        // s changes value between the start of the match and when its value is checked.
        _ if {
            *h = !*h;
            false
        } => (),
        true => (),
        false => (),
    }
}

fn bad_mutation_in_guard(mut t: bool) {
    match t {
        true => (),
        false if {
            t = true; //~ ERROR
            false
        } => (),
        false => (),
    }
}

fn bad_mutation_in_guard2(mut u: bool) {
    match u {
        // Guard changes the value bound in the last pattern.
        _ => (),
        _ if {
            u = true; //~ ERROR
            false
        } => (),
        x => (),
    }
}

pub fn bad_mutation_in_guard3(mut x: Option<Option<&i32>>) {
    // Check that nested patterns are checked.
    match x {
        None => (),
        Some(None) => (),
        _ if {
            match x {
                Some(ref mut r) => *r = None, //~ ERROR
                _ => return,
            };
            false
        } => (),
        Some(Some(r)) => println!("{}", r),
    }
}


fn bad_mutation_in_guard4(mut w: (&mut bool,)) {
    match w {
        // Guard changes the value bound in the last pattern.
        _ => (),
        _ if {
            *w.0 = true; //~ ERROR
            false
        } => (),
        x => (),
    }
}

fn bad_indirect_mutation_in_guard(mut y: &bool) {
    match *y {
        true => (),
        false if {
            y = &true; //~ ERROR
            false
        } => (),
        false => (),
    }
}

fn bad_indirect_mutation_in_guard2(mut z: &bool) {
    match z {
        &true => (),
        &false if {
            z = &true; //~ ERROR
            false
        } => (),
        &false => (),
    }
}

fn bad_indirect_mutation_in_guard3(mut a: &bool) {
    // Same as bad_indirect_mutation_in_guard2, but using match ergonomics
    match a {
        true => (),
        false if {
            a = &true; //~ ERROR
            false
        } => (),
        false => (),
    }
}

fn bad_indirect_mutation_in_guard4(mut b: &bool) {
    match b {
        &_ => (),
        &_ if {
            b = &true; //~ ERROR
            false
        } => (),
        &b => (),
    }
}

fn main() {}
