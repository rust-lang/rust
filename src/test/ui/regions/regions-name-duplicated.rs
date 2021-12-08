struct Foo<'a, 'a> {
    //~^ ERROR lifetime name `'a` declared twice
    //~| ERROR parameter `'a` is never used [E0392]
    x: &'a isize,
}

fn main() {}
