pub mod foo {
    pub fn bar() {
        println!("Hello world!");
    }
}
use foo::bar as main; //~ ERROR using an imported function as entry point
