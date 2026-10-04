fn generate_setter() {
    String::with_capacity(
    //~^ ERROR this function takes 1 argument but 3 arguments were supplied
    generate_setter,
    r#"
pub(crate) struct Person<T: Clone> {}
"#,
     r#""#,
    );
}

fn main() {}
