struct Foo<'a, 'a> { //~ ERROR lifetime name `'a` declared twice
    x: &'a isize
}

fn main() {}
