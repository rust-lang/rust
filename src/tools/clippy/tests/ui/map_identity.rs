#![warn(clippy::map_identity)]
#![allow(clippy::needless_return)]

fn main() {
    let x: [u16; 3] = [1, 2, 3];
    // should lint
    let _: Vec<_> = x.iter().map(not_identity).map(|x| return x).collect();
    //~^ map_identity
    let _: Vec<_> = x.iter().map(std::convert::identity).map(|y| y).collect();
    //~^ map_identity
    //~| map_identity
    let _: Option<u8> = Some(3).map(|x| x);
    //~^ map_identity
    let _: Result<i8, f32> = Ok(-3).map(|x| {
        //~^ map_identity
        return x;
    });
    // should not lint
    let _: Vec<_> = x.iter().map(|x| 2 * x).collect();
    let _: Vec<_> = x.iter().map(not_identity).map(|x| return x - 4).collect();
    let _: Option<u8> = None.map(|x: u8| x - 1);
    let _: Result<i8, f32> = Err(2.3).map(|x: i8| {
        return x + 3;
    });
    let _: Result<u32, u32> = Ok(1).map_err(|a| a);
    //~^ map_identity
    let _: Result<u32, u32> = Ok(1).map_err(|a: u32| a * 42);
    // : u32 guides type inference
    let _ = Ok(1).map_err(|a: u32| a);
    let _ = Ok(1).map_err(std::convert::identity::<u32>);
}

fn issue7189() {
    // should lint
    let x = [(1, 2), (3, 4)].iter().copied();
    let _ = x.clone().map(|(x, y)| (x, y));
    //~^ map_identity
    let _ = x.clone().map(|(x, y)| {
        //~^ map_identity
        return (x, y);
    });
    let _ = x.clone().map(|(x, y)| return (x, y));
    //~^ map_identity

    let y = [(1, 2, (3, (4,))), (5, 6, (7, (8,)))].iter().copied();
    let _ = y.clone().map(|(x, y, (z, (w,)))| (x, y, (z, (w,))));
    //~^ map_identity

    // should not lint
    let _ = x.clone().map(|(x, y)| (x, y, y));
    let _ = x.clone().map(|(x, _y)| (x,));
    let _ = x.clone().map(|(x, _)| (x,));
    let _ = x.clone().map(|(x, ..)| (x,));
    let _ = y.clone().map(|(x, y, (z, _))| (x, y, (z, z)));
    let _ = y
        .clone()
        .map(|(x, y, (z, _)): (i32, i32, (i32, (i32,)))| (x, y, (z, z)));
    let _ = y
        .clone()
        .map(|(x, y, (z, (w,))): (i32, i32, (i32, (i32,)))| (x, y, (z, (w,))));
}

fn not_identity(x: &u16) -> u16 {
    *x
}

fn issue11764() {
    let x = [(1, 2), (3, 4)];
    // don't lint: this is an `Iterator<Item = &(i32, i32)>`
    // match ergonomics makes the binding patterns into references
    // so that its type changes to `Iterator<Item = (&i32, &i32)>`
    let _ = x.iter().map(|(x, y)| (x, y));
    let _ = x.iter().map(|x| (x.0,)).map(|(x,)| x);

    // no match ergonomics for `(i32, i32)`
    let _ = x.iter().copied().map(|(x, y)| (x, y));
    //~^ map_identity
}

fn issue13904() {
    // don't lint: `it.next()` would not be legal as `it` is immutable
    let it = [1, 2, 3].into_iter();
    let _ = it.map(|x| x).next();

    // lint
    #[allow(unused_mut)]
    let mut it = [1, 2, 3].into_iter();
    let _ = it.map(|x| x).next();
    //~^ map_identity

    // lint
    let it = [1, 2, 3].into_iter();
    let _ = { it }.map(|x| x).next();
    //~^ map_identity
}
