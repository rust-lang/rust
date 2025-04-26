struct Empty;

trait Howness {}

impl Howness for () {
    fn how_are_you(&self -> Empty {
        Empty
    }
} //~ ERROR mismatched closing delimiter

fn main() {}
