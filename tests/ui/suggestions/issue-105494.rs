fn test1() {
    let _v: i32 = (1 as i32).to_string(); //~ ERROR mismatched types

    // won't suggestion
    let _v: i32 = (1 as i128).to_string(); //~ ERROR mismatched types

    let _v: &str = "foo".to_string(); //~ ERROR mismatched types
}

fn test2() {
    let mut path: String = "/usr".to_string();
    let folder: String = "lib".to_string();

    path = format!("{}/{}", path, folder).as_str(); //~ ERROR mismatched types

    println!("{}", &path);
}

fn main() {
    test1();
    test2();
}
