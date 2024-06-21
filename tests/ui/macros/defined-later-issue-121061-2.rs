mod demo {
    fn hello() {
        something_later!(); //~ ERROR cannot find macro `something_later`
    }

    macro_rules! something_later {
        () => {
            println!("successfully expanded!");
        };
    }
}

fn main() {}
