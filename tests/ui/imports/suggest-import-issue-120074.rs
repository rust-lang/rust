pub mod foo {
    pub mod bar {
        pub fn do_the_thing() -> usize {
            42
        }
    }
}

fn main() {
    println!("Hello, {}!", crate::bar::do_the_thing); //~ ERROR cannot find item `bar`
}
