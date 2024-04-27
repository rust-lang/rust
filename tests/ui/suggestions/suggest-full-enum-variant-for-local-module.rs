mod option {
    pub enum O<T> {
        Some(T),
        None,
    }
}

fn main() {
    let _: option::O<()> = (); //~ ERROR 9:28: 9:30: mismatched types [E0308]
}
