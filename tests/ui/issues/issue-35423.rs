//@ run-pass
fn main () {
    let x = 4;
    match x {
        ref r if *r < 0 => println!("got negative num {} < 0", r),
        e @ 1 ..= 100 => println!("got number within range [1,100] {}", e),
        _ => println!("no"),
    }
}
