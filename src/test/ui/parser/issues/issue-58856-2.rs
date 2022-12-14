struct Empty;

trait Howness {}

impl Howness for () {
    fn how_are_you(&self -> Empty {
    //~^ ERROR expected one of `)` or `,`, found `->`
    //~| ERROR method `how_are_you` is not a member of trait `Howness`
        Empty
    }
}
//~^ ERROR non-item in item list

fn main() {}
