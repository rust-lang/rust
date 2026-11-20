//@ run-pass
pub fn main() {
    let mut tups = vec![(0u8, 1u8)];

    for (n, m) in &tups {
        let _: &u8 = n;
        let _: &u8 = m;
    }

    for (n, m) in &mut tups {
        *n += 1;
        *m += 2;
    }

    assert_eq!(tups, vec![(1u8, 3u8)]);

    for (n, m) in tups {
        println!("{} {}", m, n);
    }
}
