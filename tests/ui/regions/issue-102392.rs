fn g(f: for<'a> fn(fn(&str, &'a str))) -> bool {
    f
    //~^ ERROR mismatched types
}

fn main() {}
