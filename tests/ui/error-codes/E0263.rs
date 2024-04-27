fn foo<'a, 'b, 'a>(x: &'a str, y: &'b str) {
    //~^ ERROR E0403
}

fn main() {}
