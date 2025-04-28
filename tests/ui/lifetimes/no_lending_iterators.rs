struct Data(String);

impl Iterator for Data {
    type Item = &str;
    //~^ ERROR 4:17: 4:18: associated type `Iterator::Item` is declared without lifetime parameters, so using a borrowed type for them requires that lifetime to come from the implemented type

    fn next(&mut self) -> Option<Self::Item> {
        Some(&self.0)
    }
}

trait Bar {
    type Item;
    fn poke(&mut self, item: Self::Item);
}

impl Bar for usize {
    type Item = &usize;
    //~^ ERROR 18:17: 18:18: in the trait associated type is declared without lifetime parameters, so using a borrowed type for them requires that lifetime to come from the implemented type

    fn poke(&mut self, item: Self::Item) {
        self += *item;
    }
}

impl Bar for isize {
    type Item<'a> = &'a isize;
    //~^ ERROR 27:14: 27:18: lifetime parameters or bounds on associated type `Item` do not match the trait declaration [E0195]

    fn poke(&mut self, item: Self::Item) {
        self += *item;
    }
}

fn main() {}
