fn a() {
    let mut vec = ~[~1, ~2, ~3];
    match vec {
        [~ref _a] => {
            vec[0] = ~4; //~ ERROR cannot assign to `(*vec)[]` because it is borrowed
        }
        _ => fail2!("foo")
    }
}

fn b() {
    let mut vec = ~[~1, ~2, ~3];
    match vec {
        [.._b] => {
            vec[0] = ~4; //~ ERROR cannot assign to `(*vec)[]` because it is borrowed
        }
    }
}

fn c() {
    let mut vec = ~[~1, ~2, ~3];
    match vec {
        [_a, .._b] => {
            //~^ ERROR cannot move out

            // Note: `_a` is *moved* here, but `b` is borrowing,
            // hence illegal.
            //
            // See comment in middle/borrowck/gather_loans/mod.rs
            // in the case covering these sorts of vectors.
        }
        _ => {}
    }
    let a = vec[0]; //~ ERROR use of partially moved value: `vec`
}

fn d() {
    let mut vec = ~[~1, ~2, ~3];
    match vec {
        [.._a, _b] => {
            //~^ ERROR cannot move out
        }
        _ => {}
    }
    let a = vec[0]; //~ ERROR use of partially moved value: `vec`
}

fn e() {
    let mut vec = ~[~1, ~2, ~3];
    match vec {
        [_a, _b, _c] => {}
        _ => {}
    }
    let a = vec[0]; //~ ERROR use of partially moved value: `vec`
}

fn main() {}
