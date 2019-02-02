// Test that a (partially) mutably borrowed place can be matched on, so long as
// we don't have to read any values that are mutably borrowed to determine
// which arm to take.
//
// Test that we don't allow mutating the value being matched on in a way that
// changes which patterns it matches, until we have chosen an arm.


#![feature(bind_by_move_pattern_guards)]
#![feature(nll)]

fn ok_mutation_in_guard(mut q: i32) {
    match q {
        // OK, mutation doesn't change which patterns g matches
        _ if { q = 1; false } => (),
        _ => (),
    }
}

fn ok_mutation_in_guard2(mut u: bool) {
    // OK value of u is unused before modification
    match u {
        _ => (),
        _ if {
            u = true;
            false
        } => (),
        x => (),
    }
}

fn ok_mutation_in_guard4(mut w: (&mut bool,)) {
    // OK value of u is unused before modification
    match w {
        _ => (),
        _ if {
            *w.0 = true;
            false
        } => (),
        x => (),
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
        // q doesn't match the pattern with the guard by the end of the guard.
        false if {
            q = true; //~ ERROR
            true
        } => (),
        _ => (),
    }
}

fn mutation_invalidates_previous_pattern_in_guard(mut r: bool) {
    match r {
        // r matches a previous pattern by the end of the guard.
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
    // OK value of s is unused before modification.
    match s {
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

fn bad_mutation_in_guard2(mut x: Option<Option<&i32>>) {
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

fn bad_mutation_in_guard3(mut t: bool) {
    match t {
        s if {
            t = !t; //~ ERROR
            false
        } => (), // What value should `s` have in the arm?
        _ => (),
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
