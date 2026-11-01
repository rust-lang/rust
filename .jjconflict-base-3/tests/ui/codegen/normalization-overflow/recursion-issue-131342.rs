//@ build-fail

fn main() {
    let mut items = vec![1, 2, 3, 4, 5].into_iter();
    problem_thingy(&mut items);
}

fn problem_thingy(items: &mut impl Iterator<Item = u8>) {
    let mut peeker = items.peekable();
    //~^ ERROR: reached the recursion limit while instantiating
    match peeker.peek() {
        Some(_) => (),
        None => return (),
    }
    problem_thingy(&mut peeker);
}
