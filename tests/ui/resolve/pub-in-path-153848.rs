//@ edition: 2015

pub(in a) mod aa { //~ ERROR cannot find module or crate `a` in the crate root
}
mod test {
    #[cfg(test)]
    use super::a;
}
fn main() {}
