// rustfmt-spaces_around_ranges: true

fn bar(v: &[u8]) {}

fn foo() {
    let a = vec![0; 20];
    for j in 0 ..= 20 {
        for i in 0 .. 3 {
            bar(a[i .. j]);
            bar(a[i ..]);
            bar(a[.. j]);
            bar(a[..= (j + 1)]);
        }
    }
}
