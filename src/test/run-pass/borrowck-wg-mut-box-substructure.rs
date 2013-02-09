struct Foo {
    x: ~[int]
}

fn main() {
    let x = @mut Foo { x: ~[ 1, 2, 3, 4, 5 ] };
    for x.x.each |x| {
        io::println(x.to_str());
    }
}

