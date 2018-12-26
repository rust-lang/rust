impl<'a> Drop for &'a mut isize {
    //~^ ERROR the Drop trait may only be implemented on structures
    //~^^ ERROR E0117
    fn drop(&mut self) {
        println!("kaboom");
    }
}

fn main() {
}
