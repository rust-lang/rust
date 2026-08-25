fn main() {
    let mut greeting = "Hello world!".to_string();
    let res = (|| (|| &greeting)())();

    greeting = "DEALLOCATED".to_string();
    //~^ ERROR cannot assign
    drop(greeting);

    println!("thread result: {:?}", res);
}
