// Check that the future-compat-lint for the reservation conflict is
// handled like any other lint.

// edition:2018

mod future_compat_allow {
    #![allow(mutable_borrow_reservation_conflict)]

    fn reservation_conflict() {
        let mut v = vec![0, 1, 2];
        let shared = &v;

        v.push(shared.len());
    }
}

mod future_compat_warn {
    #![warn(mutable_borrow_reservation_conflict)]

    fn reservation_conflict() {
        let mut v = vec![0, 1, 2];
        let shared = &v;

        v.push(shared.len());
        //~^ WARNING cannot borrow `v` as mutable
        //~| WARNING may become a hard error in the future
    }
}

mod future_compat_deny {
    #![deny(mutable_borrow_reservation_conflict)]

    fn reservation_conflict() {
        let mut v = vec![0, 1, 2];
        let shared = &v;

        v.push(shared.len());
        //~^ ERROR cannot borrow `v` as mutable
        //~| WARNING may become a hard error in the future
    }
}

fn main() {}
