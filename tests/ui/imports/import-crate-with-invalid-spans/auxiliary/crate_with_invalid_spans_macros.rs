macro_rules! add1 {
    ($e:expr) => ({
        let a = 1 + $e;
        let b = $e + 1;
        a + b - 1
    })
}
