mod option {
    pub enum O<T> {
        Some(T),
        None,
    }
}

fn main() {
    let _: option::O<()> = (); //~ ERROR mismatched types [E0308]
}
