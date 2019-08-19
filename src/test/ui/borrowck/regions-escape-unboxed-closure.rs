fn with_int(f: &mut dyn FnMut(&isize)) {
}

fn main() {
    let mut x: Option<&isize> = None;
    with_int(&mut |y| x = Some(y));
    //~^ ERROR borrowed data cannot be stored outside of its closure
}
