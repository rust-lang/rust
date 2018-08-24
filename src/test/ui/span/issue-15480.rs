fn id<T>(x: T) -> T { x }

fn main() {
    let v = vec![
        &id(3)
    ];
    //~^^ ERROR borrowed value does not live long enough

    for &&x in &v {
        println!("{}", x + 3);
    }
}
