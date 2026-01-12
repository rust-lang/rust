// We check item bounds on projection predicate's term now for well-formedness.

fn problem_thingy(items: &mut impl Iterator<Item = str>) {
    //~^ ERROR: the size for values of type `str` cannot be known at compilation time [E0277]
    items.peekable();
}

fn main() {}
