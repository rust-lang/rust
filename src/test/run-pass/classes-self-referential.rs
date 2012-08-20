struct kitten {
    let cat: Option<cat>;
    new(cat: Option<cat>) {
       self.cat = cat;
    }
}

type cat = @kitten;

fn main() {}
