#![feature(trait_alias)]

trait Foo = std::io::Read + std::io::Write;
trait Bar = Foo;

fn main() {
    let _: Box<std::io::Read + std::io::Write>;
    //~^ ERROR only auto traits can be used as additional traits in a trait object [E0225]
    let _: Box<Bar>;
    //~^ ERROR only auto traits can be used as additional traits in a trait object [E0225]
}
