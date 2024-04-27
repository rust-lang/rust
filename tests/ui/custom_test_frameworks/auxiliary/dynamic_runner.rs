use std::process::exit;

pub trait Testable {
    // Name of the test
    fn name(&self) -> String;

    // Tests pass by default
    fn run(&self) -> bool {
        true
    }

    // A test can generate subtests
    fn subtests(&self) -> Vec<Box<dyn Testable>> {
        vec![]
    }
}

fn run_test(t: &dyn Testable) -> bool {
    let success = t.subtests().into_iter().all(|sub_t| run_test(&*sub_t)) && t.run();
    println!("{}...{}", t.name(), if success { "SUCCESS" } else { "FAIL" });
    success
}

pub fn runner(tests: &[&dyn Testable]) {
    let mut failed = false;
    for t in tests {
        if !run_test(*t) {
            failed = true;
        }
    }

    if failed {
        exit(1);
    }
}
