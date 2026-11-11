const NUM_RUNS: usize = 10;

fn run_self(exe: &str) -> usize {
    use std::process::Command;
    let mut set = std::collections::HashSet::new();

    let mut cmd = Command::new(exe);
    cmd.arg("--report");
    (0..NUM_RUNS).for_each(|_| {
        set.insert(cmd.output().expect("failed to execute process").stdout);
    });
    set.len()
}

fn main() {
    let mut args = std::env::args();
    let arg0 = args.next().unwrap();
    match args.next() {
        Some(s) if s.eq("--report") => {
            println!("main = {:#?}", main as fn() as usize);
        }
        Some(s) if s.eq("--test-no-aslr") => {
            let cnt = run_self(&arg0);
            if cnt != 1 {
                eprintln!("FAIL: {} most likely ASLR", arg0);
                std::process::exit(1);
            }
            println!("PASS: {} does no ASLR", arg0);
        }
        Some(s) if s.eq("--test-aslr") => {
            let cnt = run_self(&arg0);
            if cnt == 1 {
                eprintln!("FAIL: {} most likely no ASLR", arg0);
                std::process::exit(1);
            }
            println!("PASS: {} does ASLR", arg0);
        }
        Some(_) | None => {
            println!("Usage: {} --test-no-aslr | --test-aslr", arg0);
            std::process::exit(1);
        }
    }
}
