use std::path::Path;

fn main() {
    let path = Path::new("/tmp/foo/bar.txt");
    println!("{}", path);
    //~^ ERROR E0277
}
