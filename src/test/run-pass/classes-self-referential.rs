struct kitten {
    let cat: Option<cat>;
}

fn kitten(cat: Option<cat>) -> kitten {
    kitten {
        cat: cat
    }
}

type cat = @kitten;

fn main() {}
