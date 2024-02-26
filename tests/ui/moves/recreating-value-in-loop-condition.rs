fn iter<T>(vec: Vec<T>) -> impl Iterator<Item = T> {
    vec.into_iter()
}
fn foo() {
    let vec = vec!["one", "two", "three"];
    while let Some(item) = iter(vec).next() { //~ ERROR use of moved value
        println!("{:?}", item);
    }
}
fn bar() {
    let vec = vec!["one", "two", "three"];
    loop {
        let Some(item) = iter(vec).next() else { //~ ERROR use of moved value
            break;
        };
        println!("{:?}", item);
    }
}
fn baz() {
    let vec = vec!["one", "two", "three"];
    loop {
        let item = iter(vec).next(); //~ ERROR use of moved value
        if item.is_none() {
            break;
        }
        println!("{:?}", item);
    }
}
fn qux() {
    let vec = vec!["one", "two", "three"];
    loop {
        if let Some(item) = iter(vec).next() { //~ ERROR use of moved value
            println!("{:?}", item);
        }
    }
}
fn zap() {
    loop {
        let vec = vec!["one", "two", "three"];
        loop {
            loop {
                loop {
                    if let Some(item) = iter(vec).next() { //~ ERROR use of moved value
                        println!("{:?}", item);
                    }
                }
            }
        }
    }
}
fn main() {
    foo();
    bar();
    baz();
    qux();
    zap();
}
