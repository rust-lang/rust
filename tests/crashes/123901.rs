//@ known-bug: #123901
//@ edition:2021

pub fn test(test: &u64, temp: &u64) {
    async |check, a, b| {
        temp.abs_diff(12);
    };
}
