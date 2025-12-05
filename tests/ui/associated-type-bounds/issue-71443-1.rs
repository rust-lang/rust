struct Incorrect;

fn hello<F: for<'a> Iterator<Item: 'a>>() {
    Incorrect //~ERROR: mismatched types
}

fn main() {}
