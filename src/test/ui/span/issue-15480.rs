fn id<T>(x: T) -> T { x }

fn main() {
    let v = vec![
        &id(3)
    ];
    //~^^ ERROR temporary value dropped while borrowed

    for &&x in &v {
        println!("{}", x + 3);
    }
}
