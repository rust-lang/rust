// Define a module named `my_module`
mod my_module {
    // Import the `io` module from the standard library
    use std::io;

    // Define a function named `print_input`
    pub fn print_input() {
        println!("Enter some text:");

        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");

        println!("You entered: {}", input);
    }
}

// Define the entry point of the program
fn main() {
    // Call the `print_input` function from the `my_module` module
    my_module::print_input();
}
