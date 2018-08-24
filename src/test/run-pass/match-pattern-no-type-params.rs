enum maybe<T> { nothing, just(T), }

fn foo(x: maybe<isize>) {
    match x {
        maybe::nothing => { println!("A"); }
        maybe::just(_a) => { println!("B"); }
    }
}

pub fn main() { }
