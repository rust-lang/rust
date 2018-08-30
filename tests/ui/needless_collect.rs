#[warn(clippy, needless_collect)]
#[allow(unused_variables, iter_cloned_collect)]
fn main() {
    let sample = [1; 5];
    let len = sample.iter().collect::<Vec<_>>().len();
    if sample.iter().collect::<Vec<_>>().is_empty() {
        // Empty
    }
    sample.iter().cloned().collect::<Vec<_>>().contains(&1);
}
