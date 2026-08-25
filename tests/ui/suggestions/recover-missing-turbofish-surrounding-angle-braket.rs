fn main() {
    let _ = vec![1, 2, 3].into_iter().collect::Vec<_>();
    //~^ ERROR generic parameters without surrounding angle brackets
    let _ = vec![1, 2, 3].into_iter().collect::Vec<_>>>>();
    //~^ ERROR generic parameters without surrounding angle brackets
    let _ = vec![1, 2, 3].into_iter().collect::Vec<_>>>();
    //~^ ERROR generic parameters without surrounding angle brackets
    let _ = vec![1, 2, 3].into_iter().collect::Vec<_>>();
    //~^ ERROR generic parameters without surrounding angle brackets
}
