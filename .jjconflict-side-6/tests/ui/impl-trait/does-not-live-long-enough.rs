struct List {
    data: Vec<String>,
}
impl List {
    fn started_with<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item=&'a str> {
        self.data.iter().filter(|s| s.starts_with(prefix)).map(|s| s.as_ref())
        //~^ ERROR E0373
    }
}

fn main() {}
