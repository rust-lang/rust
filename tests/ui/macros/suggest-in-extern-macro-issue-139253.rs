#[derive(Clone, Debug)]
struct Point<T> {
    v: T,
}

macro_rules! local_macro {
    ($val:expr $(,)?) => {
        match $val {
            tmp => tmp, //~ ERROR mismatched types [E0308]
        }
    };
}

fn main() {
    let a: Point<u8> = dbg!(Point { v: 42 });
    let b: Point<u8> = dbg!(&a); //~ ERROR mismatched types [E0308]
    let c: Point<u8> = local_macro!(&a);
}
