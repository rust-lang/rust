fn iter<T>(vec: Vec<T>) -> impl Iterator<Item = T> {
    vec.into_iter()
}
fn foo() {
    let vec = vec!["one", "two", "three"];
    while let Some(item) = iter(vec).next() { //~ ERROR use of moved value
    //~^ HELP consider moving the expression out of the loop so it is only moved once
        println!("{:?}", item);
    }
}
fn bar() {
    let vec = vec!["one", "two", "three"];
    loop {
    //~^ HELP consider moving the expression out of the loop so it is only moved once
        let Some(item) = iter(vec).next() else { //~ ERROR use of moved value
            break;
        };
        println!("{:?}", item);
    }
}
fn baz() {
    let vec = vec!["one", "two", "three"];
    loop {
    //~^ HELP consider moving the expression out of the loop so it is only moved once
        let item = iter(vec).next(); //~ ERROR use of moved value
        //~^ HELP consider cloning
        if item.is_none() {
            break;
        }
        println!("{:?}", item);
    }
}
fn qux() {
    let vec = vec!["one", "two", "three"];
    loop {
    //~^ HELP consider moving the expression out of the loop so it is only moved once
        if let Some(item) = iter(vec).next() { //~ ERROR use of moved value
            println!("{:?}", item);
            break;
        }
    }
}
fn zap() {
    loop {
        let vec = vec!["one", "two", "three"];
        loop {
        //~^ HELP consider moving the expression out of the loop so it is only moved once
            loop {
                loop {
                    if let Some(item) = iter(vec).next() { //~ ERROR use of moved value
                        println!("{:?}", item);
                        break;
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
