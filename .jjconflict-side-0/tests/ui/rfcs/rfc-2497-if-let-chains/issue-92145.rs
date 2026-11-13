//@ check-pass
//@ edition:2024

fn main() {
    let opt = Some("foo bar");

    if true && let Some(x) = opt {
        println!("{}", x);
    }
}
