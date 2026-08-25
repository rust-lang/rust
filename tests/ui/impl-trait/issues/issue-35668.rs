fn func<'a, T>(a: &'a [T]) -> impl Iterator<Item=&'a T> {
    a.iter().map(|a| a*a)
    //~^ ERROR cannot multiply `&T` by `&T`
}

fn main() {
    let a = (0..30).collect::<Vec<_>>();

    for k in func(&a) {
        println!("{}", k);
    }
}
