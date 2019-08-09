#![feature(trait_alias)]

trait Foo = std::io::Read + std::io::Write;

fn main() {
    let _: Box<dyn std::io::Read + std::io::Write>;
    //~^ ERROR only auto traits can be used as additional traits in a trait object [E0225]
    let _: Box<dyn Foo>;
    //~^ ERROR only auto traits can be used as additional traits in a trait object [E0225]
}
