struct Data(String);

impl Iterator for Data {
    type Item = &str;
    //~^ ERROR 4:17: 4:18: associated type `Iterator::Item` is declared without lifetime parameters, so using a borrowed type for them requires that lifetime to come from the implemented type.

    fn next(&mut self) -> Option<Self::Item> {
        Some(&self.0)
    }
}

fn main() {}
