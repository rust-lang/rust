struct List {
    data: Vec<String>,
}
impl List {
    fn started_with<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item=&'a str> {
        self.data.iter().filter(|s| s.starts_with(prefix)).map(|s| s.as_ref())
        //~^ ERROR does not live long enough
    }
}

fn main() {}
