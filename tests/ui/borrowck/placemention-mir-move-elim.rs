fn main() {
    let x: String;
    let _ = x; //~ ERROR
    x = "foo".to_owned();
    let _ = x; // OK
    drop(x);
    let _ = x; //~ ERROR

    let x: (String,);
    let _ = x; //~ ERROR
    x = ("foo".to_owned(),);
    let _ = x; // OK
    drop(x.0);
    let _ = x; // OK
}
