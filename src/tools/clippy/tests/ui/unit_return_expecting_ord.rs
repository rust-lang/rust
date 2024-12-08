#![warn(clippy::unit_return_expecting_ord)]
#![allow(clippy::needless_return)]
#![allow(clippy::unused_unit)]
#![allow(clippy::useless_vec)]

struct Struct {
    field: isize,
}

fn double(i: isize) -> isize {
    i * 2
}

fn unit(_i: isize) {}

fn main() {
    let mut structs = vec![Struct { field: 2 }, Struct { field: 1 }];
    structs.sort_by_key(|s| {
        //~^ ERROR: this closure returns the unit type which also implements Ord
        double(s.field);
    });
    structs.sort_by_key(|s| double(s.field));
    structs.is_sorted_by_key(|s| {
        //~^ ERROR: this closure returns the unit type which also implements PartialOrd
        double(s.field);
    });
    structs.is_sorted_by_key(|s| {
        //~^ ERROR: this closure returns the unit type which also implements PartialOrd
        if s.field > 0 {
            ()
        } else {
            return ();
        }
    });
    structs.sort_by_key(|s| {
        return double(s.field);
    });
    structs.sort_by_key(|s| unit(s.field));
    //~^ ERROR: this closure returns the unit type which also implements Ord
}
