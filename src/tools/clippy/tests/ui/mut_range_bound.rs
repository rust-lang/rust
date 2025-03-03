#![allow(unused)]

fn main() {}

fn mut_range_bound_upper() {
    let mut m = 4;
    for i in 0..m {
        m = 5;
        //~^ mut_range_bound
    }
}

fn mut_range_bound_lower() {
    let mut m = 4;
    for i in m..10 {
        m *= 2;
        //~^ mut_range_bound
    }
}

fn mut_range_bound_both() {
    let mut m = 4;
    let mut n = 6;
    for i in m..n {
        m = 5;
        //~^ mut_range_bound

        n = 7;
        //~^ mut_range_bound
    }
}

fn mut_range_bound_no_mutation() {
    let mut m = 4;
    for i in 0..m {
        continue;
    } // no warning
}

fn mut_borrow_range_bound() {
    let mut m = 4;
    for i in 0..m {
        let n = &mut m;
        //~^ mut_range_bound

        *n += 1;
    }
}

fn immut_borrow_range_bound() {
    let mut m = 4;
    for i in 0..m {
        let n = &m;
    }
}

fn immut_range_bound() {
    let m = 4;
    for i in 0..m {
        continue;
    } // no warning
}

fn mut_range_bound_break() {
    let mut m = 4;
    for i in 0..m {
        if m == 4 {
            m = 5; // no warning because of immediate break
            break;
        }
    }
}

fn mut_range_bound_no_immediate_break() {
    let mut m = 4;
    for i in 0..m {
        // warning because it is not immediately followed by break
        m = 2;
        //~^ mut_range_bound

        if m == 4 {
            break;
        }
    }

    let mut n = 3;
    for i in n..10 {
        if n == 4 {
            // FIXME: warning because it is not immediately followed by break
            n = 1;
            //~^ mut_range_bound

            let _ = 2;
            break;
        }
    }
}
