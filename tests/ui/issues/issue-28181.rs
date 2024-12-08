//@ run-pass
fn bar<F>(f: F) -> usize where F: Fn([usize; 1]) -> usize { f([2]) }

fn main() {
    bar(|u| { u[0] });
}
