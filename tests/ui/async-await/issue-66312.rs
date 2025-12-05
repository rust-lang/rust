//@ edition:2018

trait Test<T> {
    fn is_some(self: T); //~ ERROR invalid `self` parameter type
}

async fn f() {
    let x = Some(2);
    if x.is_some() { //~ ERROR mismatched types
        println!("Some");
    }
}

fn main() {}
