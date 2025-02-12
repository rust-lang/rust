#![warn(clippy::inspect_for_each)]

fn main() {
    let a: Vec<usize> = vec![1, 2, 3, 4, 5];

    let mut b: Vec<usize> = Vec::new();
    a.into_iter().inspect(|x| assert!(*x > 0)).for_each(|x| {
        //~^ inspect_for_each

        let y = do_some(x);
        let z = do_more(y);
        b.push(z);
    });

    assert_eq!(b, vec![4, 5, 6, 7, 8]);
}

fn do_some(a: usize) -> usize {
    a + 1
}

fn do_more(a: usize) -> usize {
    a + 2
}
