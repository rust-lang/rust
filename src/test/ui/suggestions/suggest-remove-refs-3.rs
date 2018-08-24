fn main() {
    let v = vec![0, 1, 2, 3];

    for (i, n) in & & &
        & &v
        .iter()
        .enumerate() {
        //~^^^^ ERROR the trait bound
        println!("{}", i);
    }
}
