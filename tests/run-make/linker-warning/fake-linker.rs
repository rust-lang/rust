fn main() {
    for arg in std::env::args() {
        match &*arg {
            "run_make_info" => println!("foo"),
            "run_make_warn" => eprintln!("warning: bar"),
            "run_make_error" => {
                eprintln!("error: baz");
                std::process::exit(1);
            }
            _ => (),
        }
    }
}
