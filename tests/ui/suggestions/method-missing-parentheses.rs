fn main() {
    let _ = vec![].into_iter().collect::<usize>;
    //~^ ERROR attempted to take value of method `collect` on type `std::vec::IntoIter<_>`
}
