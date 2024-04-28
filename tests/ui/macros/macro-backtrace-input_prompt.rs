#[macro_export]
macro_rules! input_prompt {
    ($prompt:expr) => {{
        use std::io::{stdin, stdout, Write};
        print!("{}", $prompt);
        let _ = stdout().flush();
        let mut input = String::new();
        stdin().read_line(&mut input).expect("Failed to read input");
        input.trim().to_string() // Changed parse() to to_string() to handle strings
    }};
}    
fn main (){
    let name = input_prompt!("Enter Your Name");
    println!("my name is{}",name);
}