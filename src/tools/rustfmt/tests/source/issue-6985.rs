macro_rules ! say_hello {
    () => {
        println!("Hello!")
    };
}

macro_rules /* comment */ ! say_goodbye {
    () => {
        println!("Goodbye!")
    };
}

macro_rules // comment
! do_nothing {
    () => {};
}

fn main() {
    say_hello!();
    say_goodbye!()
}
