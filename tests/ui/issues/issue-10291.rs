fn test<'x>(x: &'x isize) {
    drop::<Box<dyn for<'z> FnMut(&'z isize) -> &'z isize>>(Box::new(|z| {
        x
        //~^ ERROR lifetime may not live long enough
    }));
}

fn main() {}
