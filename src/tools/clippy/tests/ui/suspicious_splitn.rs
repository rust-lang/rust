#![warn(clippy::suspicious_splitn)]
#![allow(clippy::needless_splitn)]

fn main() {
    let _ = "a,b,c".splitn(3, ',');
    let _ = [0, 1, 2, 1, 3].splitn(3, |&x| x == 1);
    let _ = "".splitn(0, ',');
    let _ = [].splitn(0, |&x: &u32| x == 1);

    let _ = "a,b".splitn(0, ',');
    //~^ ERROR: `splitn` called with `0` splits
    //~| NOTE: the resulting iterator will always return `None`
    let _ = "a,b".rsplitn(0, ',');
    //~^ ERROR: `rsplitn` called with `0` splits
    //~| NOTE: the resulting iterator will always return `None`
    let _ = "a,b".splitn(1, ',');
    //~^ ERROR: `splitn` called with `1` split
    //~| NOTE: the resulting iterator will always return the entire string followed by `No
    let _ = [0, 1, 2].splitn(0, |&x| x == 1);
    //~^ ERROR: `splitn` called with `0` splits
    //~| NOTE: the resulting iterator will always return `None`
    let _ = [0, 1, 2].splitn_mut(0, |&x| x == 1);
    //~^ ERROR: `splitn_mut` called with `0` splits
    //~| NOTE: the resulting iterator will always return `None`
    let _ = [0, 1, 2].splitn(1, |&x| x == 1);
    //~^ ERROR: `splitn` called with `1` split
    //~| NOTE: the resulting iterator will always return the entire slice followed by `Non
    let _ = [0, 1, 2].rsplitn_mut(1, |&x| x == 1);
    //~^ ERROR: `rsplitn_mut` called with `1` split
    //~| NOTE: the resulting iterator will always return the entire slice followed by `Non

    const X: usize = 0;
    let _ = "a,b".splitn(X + 1, ',');
    //~^ ERROR: `splitn` called with `1` split
    //~| NOTE: the resulting iterator will always return the entire string followed by `No
    let _ = "a,b".splitn(X, ',');
    //~^ ERROR: `splitn` called with `0` splits
    //~| NOTE: the resulting iterator will always return `None`
}
