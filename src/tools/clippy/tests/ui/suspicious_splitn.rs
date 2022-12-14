#![warn(clippy::suspicious_splitn)]
#![allow(clippy::needless_splitn)]

fn main() {
    let _ = "a,b,c".splitn(3, ',');
    let _ = [0, 1, 2, 1, 3].splitn(3, |&x| x == 1);
    let _ = "".splitn(0, ',');
    let _ = [].splitn(0, |&x: &u32| x == 1);

    let _ = "a,b".splitn(0, ',');
    let _ = "a,b".rsplitn(0, ',');
    let _ = "a,b".splitn(1, ',');
    let _ = [0, 1, 2].splitn(0, |&x| x == 1);
    let _ = [0, 1, 2].splitn_mut(0, |&x| x == 1);
    let _ = [0, 1, 2].splitn(1, |&x| x == 1);
    let _ = [0, 1, 2].rsplitn_mut(1, |&x| x == 1);

    const X: usize = 0;
    let _ = "a,b".splitn(X + 1, ',');
    let _ = "a,b".splitn(X, ',');
}
