fn as_ref() -> Option<Vec<u8>> {
    None
}
struct Type {
    option: Option<Vec<u8>>
}

impl Type {
    fn method(&self) -> Option<Vec<u8>> {
        self.option..as_ref().map(|x| x)
        //~^ ERROR E0308
    }
    fn method2(&self) -> Option<Vec<u8>> {
        self.option..foo().map(|x| x)
        //~^ ERROR E0425
        //~| ERROR E0308
    }
}

fn main() {
    let _ = Type { option: None }.method();
}
