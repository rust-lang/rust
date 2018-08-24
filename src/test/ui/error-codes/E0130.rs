extern {
    fn foo((a, b): (u32, u32));
    //~^ ERROR E0130
}

fn main() {
}
