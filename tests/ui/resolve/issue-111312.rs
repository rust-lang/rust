//@ edition: 2021

trait Has {
    fn has() {}
}

trait HasNot {}

fn main() {
    <dyn HasNot>::has();
    //~^ ERROR no function or associated item named `has` found for trait `HasNot`
}
