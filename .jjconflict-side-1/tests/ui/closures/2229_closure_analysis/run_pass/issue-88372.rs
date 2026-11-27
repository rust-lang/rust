//@ edition:2021
//@ run-pass


fn solve<F>(validate: F) -> Option<u64>
where
    F: Fn(&mut [i8; 1]),
{
    let mut position: [i8; 1] = [1];
    Some(0).map(|_| {
        validate(&mut position);
        let [_x] = position;
        0
    })
}

fn main() {
    solve(|_| ());
}
