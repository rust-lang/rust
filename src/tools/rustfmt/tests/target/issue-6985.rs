macro_rules! say_hello {
    () => {
        println!("Hello!")
    };
}

macro_rules! say_goodbye {
    () => {
        println!("Goodbye!")
    };
}

macro_rules! do_nothing {
    () => {};
}

fn main() {
    say_hello!();
    say_goodbye!()
}
