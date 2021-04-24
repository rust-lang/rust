// run-rustfix
#![warn(clippy::cloned_instead_of_copied)]

fn main() {
    // yay
    let _ = [1].iter().cloned();
    let _ = vec!["hi"].iter().cloned();
    let _ = Some(&1).cloned();
    let _ = Box::new([1].iter()).cloned();
    let _ = Box::new(Some(&1)).cloned();

    // nay
    let _ = [String::new()].iter().cloned();
    let _ = Some(&String::new()).cloned();
}
