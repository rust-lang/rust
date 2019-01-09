fn test<'x>(x: &'x isize) {
    drop::<Box<for<'z> FnMut(&'z isize) -> &'z isize>>(Box::new(|z| {
        x //~ ERROR E0312
    }));
}

fn main() {}
