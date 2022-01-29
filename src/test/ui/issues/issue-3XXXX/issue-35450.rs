macro_rules! m { ($($t:tt)*) => { $($t)* } }

fn main() {
    m!($t); //~ ERROR expected expression
}
