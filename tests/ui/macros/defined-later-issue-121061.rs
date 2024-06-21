fn main() {
    something_later!(); //~ ERROR cannot find macro `something_later`
}

macro_rules! something_later {
    () => {
        println!("successfully expanded!");
    };
}
