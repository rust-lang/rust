pub trait Testable {
    fn name(&self) -> String;
    fn run(&self) -> Option<String>; // None will be success, Some is the error message
}

pub fn runner(tests: &[&dyn Testable]) {
    for t in tests {
        print!("{}........{}", t.name(), t.run().unwrap_or_else(|| "SUCCESS".to_string()));
    }
}
