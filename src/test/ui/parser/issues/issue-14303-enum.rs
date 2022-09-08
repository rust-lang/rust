enum X<'a, T, 'b> {
//~^ ERROR lifetime parameters must be declared prior to type or const parameters
    A(&'a &'b T)
}

fn main() {}
