mod demo {
    fn hello() {
        something_later!(); //~ ERROR cannot find macro `something_later` in this scope
    }

    macro_rules! something_later {
        () => {
            println!("successfully expanded!");
        };
    }
}

fn main() {}
