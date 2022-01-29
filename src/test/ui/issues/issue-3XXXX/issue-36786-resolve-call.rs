// run-pass
// Ensure that types that rely on obligations are autoderefed
// correctly

fn main() {
    let x : Vec<Box<dyn Fn()>> = vec![Box::new(|| ())];
    x[0]()
}
