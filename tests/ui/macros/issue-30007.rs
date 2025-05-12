macro_rules! t {
    () => ( String ; );     //~ ERROR macro expansion ignores `;`
}

fn main() {
    let i: Vec<t!()>;
}
