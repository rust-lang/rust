fn test<'x>(x: &'x isize) {
    drop::<Box<dyn for<'z> FnMut(&'z isize) -> &'z isize>>(Box::new(|z| {
        x //~ ERROR E0312
    }));
}

fn main() {}
