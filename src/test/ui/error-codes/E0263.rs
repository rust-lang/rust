fn foo<'a, 'b, 'a>(x: &'a str, y: &'b str) {
    //~^ ERROR E0263
}

fn main() {}
