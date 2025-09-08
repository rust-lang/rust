fn no_format_specifier_two_unused_args() {
    println!("Hello", "World");
    //~^ ERROR argument never used
    //~| NOTE formatting specifier missing
    //~| NOTE argument never used
    //~| HELP format specifiers use curly braces, consider adding a format specifier
}

fn no_format_specifier_multiple_unused_args() {
    println!("list: ", 1, 2, 3);
    //~^ ERROR multiple unused formatting arguments
    //~| NOTE multiple missing formatting specifiers
    //~| NOTE argument never used
    //~| NOTE argument never used
    //~| NOTE argument never used
    //~| HELP format specifiers use curly braces, consider adding 3 format specifiers
}

fn missing_format_specifiers_one_unused_arg() {
    println!("list: {}, {}", 1, 2, 3);
    //~^ ERROR argument never used
    //~| NOTE formatting specifier missing
    //~| NOTE argument never used
}

fn missing_format_specifiers_multiple_unused_args() {
    println!("list: {}", 1, 2, 3);
    //~^ ERROR multiple unused formatting arguments
    //~| NOTE multiple missing formatting specifiers
    //~| NOTE argument never used
    //~| NOTE argument never used
    //~| NOTE consider adding 2 format specifiers
}

fn main() { }
