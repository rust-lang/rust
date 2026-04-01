struct T(String);

impl PartialEq<String> for T {
    fn eq(&self, other: &String) -> bool {
        &self.0 == other
    }
}

fn main() {
    String::from("Girls Band Cry") == T(String::from("Girls Band Cry")); //~ ERROR can't compare `String` with `T` [E0277]
}
