use std::io::{self, Write};
pub mod git_hook;
pub mod intellij;

/// This function will asked the user the given question and wait for user input
/// either `true` for yes and `false` for no.
fn ask_yes_no_question(question: &str) -> bool {
    // This code was proudly stolen from rusts bootstrapping tool.

    fn ask_with_result(question: &str) -> io::Result<bool> {
        let mut input = String::new();
        Ok(loop {
            print!("{}: [y/N] ", question);
            io::stdout().flush()?;
            input.clear();
            io::stdin().read_line(&mut input)?;
            break match input.trim().to_lowercase().as_str() {
                "y" | "yes" => true,
                "n" | "no" | "" => false,
                _ => {
                    println!("error: unrecognized option '{}'", input.trim());
                    println!("note: press Ctrl+C to exit");
                    continue;
                },
            };
        })
    }

    ask_with_result(question).unwrap_or_default()
}
