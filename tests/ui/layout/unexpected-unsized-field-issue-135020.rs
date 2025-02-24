//@ check-pass

fn problem_thingy(items: &mut impl Iterator<Item = str>) {
    items.peekable();
}

fn main() {}
