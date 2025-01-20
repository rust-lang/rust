pub fn my_func(a: i32, b: i32) -> i32 {
    let mut res = a;
    for i in a..b {
        res += i;
    }
    res
}
