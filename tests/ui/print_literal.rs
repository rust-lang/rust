#![warn(clippy::print_literal)]

fn main() {
    // these should be fine
    print!("Hello");
    println!("Hello");
    let world = "world";
    println!("Hello {}", world);
    println!("Hello {world}", world = world);
    println!("3 in hex is {:X}", 3);
    println!("2 + 1 = {:.4}", 3);
    println!("2 + 1 = {:5.4}", 3);
    println!("Debug test {:?}", "hello, world");
    println!("{0:8} {1:>8}", "hello", "world");
    println!("{1:8} {0:>8}", "hello", "world");
    println!("{foo:8} {bar:>8}", foo = "hello", bar = "world");
    println!("{bar:8} {foo:>8}", foo = "hello", bar = "world");
    println!("{number:>width$}", number = 1, width = 6);
    println!("{number:>0width$}", number = 1, width = 6);
    println!("{} of {:b} people know binary, the other half doesn't", 1, 2);
    println!("10 / 4 is {}", 2.5);
    println!("2 + 1 = {}", 3);
    println!("From expansion {}", stringify!(not a string literal));

    // these should throw warnings
    print!("Hello {}", "world");
    println!("Hello {} {}", world, "world");
    println!("Hello {}", "world");
    println!("{} {:.4}", "a literal", 5);

    // positional args don't change the fact
    // that we're using a literal -- this should
    // throw a warning
    println!("{0} {1}", "hello", "world");
    println!("{1} {0}", "hello", "world");

    // named args shouldn't change anything either
    println!("{foo} {bar}", foo = "hello", bar = "world");
    println!("{bar} {foo}", foo = "hello", bar = "world");
}
