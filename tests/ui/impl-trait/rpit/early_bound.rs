use std::convert::identity;

fn test<'a: 'a>(n: bool) -> impl Sized + 'a {
    let true = n else { loop {} };
    let _ = || {
        let _ = identity::<&'a ()>(test(false));
        //~^ ERROR concrete type differs from previous defining opaque type use
    };
    loop {}
}

fn main() {}
