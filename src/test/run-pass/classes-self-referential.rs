struct kitten {
    let cat: option<cat>;
    new(cat: option<cat>) {
       self.cat = cat;
    }
}

type cat = @kitten;

fn main() {}
