

#![warn(print_literal)]

fn main() {
    // these should be fine
    print!("Hello");
    println!("Hello");
    let world = "world";
    println!("Hello {}", world);
    println!("3 in hex is {:X}", 3);

    // this in theory shouldn't yield a warning,
    // but at present time, it's a known edge case
    // that isn't handled (because we can't expand
    // `println!` and not `env!`)
    println!("foo: {}", env!("BAR"));

    // these should throw warnings
    print!("Hello {}", "world");
    println!("Hello {} {}", world, "world");
    println!("Hello {}", "world");
    println!("10 / 4 is {}", 2.5);
    println!("2 + 1 = {}", 3);
    println!("2 + 1 = {:.4}", 3);
    println!("2 + 1 = {:5.4}", 3);
    println!("Debug test {:?}", "hello, world");

    // positional args don't change the fact
    // that we're using a literal -- this should
    // throw a warning
    println!("{0} {1}", "hello", "world");
    println!("{1} {0}", "hello", "world");

    // named args shouldn't change anything either
    println!("{foo} {bar}", foo="hello", bar="world");
    println!("{bar} {foo}", foo="hello", bar="world");
}
