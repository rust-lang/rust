#![feature(nll)]

// Check that we assert that pointers have a common subtype for comparisons

fn compare_const<'a, 'b>(x: *const &mut &'a i32, y: *const &mut &'b i32) {
    x == y;
    //~^ ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
}

fn compare_mut<'a, 'b>(x: *mut &'a i32, y: *mut &'b i32) {
    x == y;
    //~^ ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
}

fn compare_fn_ptr<'a, 'b, 'c>(f: fn(&'c mut &'a i32), g: fn(&'c mut &'b i32)) {
    f == g;
    //~^ ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
}

fn compare_hr_fn_ptr<'a>(f: fn(&'a i32), g: fn(&i32)) {
    f == g;
    //~^ ERROR higher-ranked subtype error
}

fn compare_const_fn_ptr<'a>(f: *const fn(&'a i32), g: *const fn(&i32)) {
    f == g;
    //~^ ERROR higher-ranked subtype error
}

fn main() {}
