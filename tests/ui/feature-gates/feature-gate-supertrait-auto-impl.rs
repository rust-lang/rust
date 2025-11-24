trait Supertrait {}
trait Subtrait: Supertrait {
    auto impl Supertrait;
    //~^ ERROR feature is under construction
}

fn main() {}
