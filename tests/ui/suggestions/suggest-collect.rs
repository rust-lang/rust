fn main() {
    let _x: String = "hello".chars().map(|c| c);
    //~^ ERROR mismatched types
    //~| HELP consider using `.collect()` to convert the `Iterator` into a `String`

    let _y: Vec<i32> = vec![1, 2, 3].into_iter().map(|x| x);
    //~^ ERROR mismatched types
    //~| HELP consider using `.collect()` to convert the `Iterator` into a `Vec<i32>`

    let res: Result<Vec<i32>, _> = ["1", "2"].into_iter().map(|s| s.parse::<i32>());
    //~^ ERROR mismatched types
    //~| HELP consider using `.collect()` to convert the `Iterator` into a `Result<Vec<i32>, _>`
    let (a, b): (Vec<i32>, Vec<i32>) = vec![1, 2].into_iter().map(|x| (x, x));
    //~^ ERROR mismatched types
    //~| HELP consider using `.collect()` to convert the `Iterator` into a `(Vec<i32>, Vec<i32>)`
}
