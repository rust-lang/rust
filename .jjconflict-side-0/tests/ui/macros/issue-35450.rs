macro_rules! m { ($($t:tt)*) => { $($t)* } }

fn main() {
    m!($t); //~ ERROR cannot find macro parameter `$t` in this scope
}
