impl<T> T { //~ ERROR E0118
    fn get_state(&self) -> String {
       String::new()
    }
}

fn main() {}
