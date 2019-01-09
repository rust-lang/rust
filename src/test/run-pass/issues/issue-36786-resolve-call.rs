// run-pass
// Ensure that types that rely on obligations are autoderefed
// correctly

fn main() {
    let x : Vec<Box<Fn()>> = vec![Box::new(|| ())];
    x[0]()
}
