macro_rules! t {
    () => ( String ; );     //~ ERROR macro expansion ignores token `;`
}

fn main() {
    let i: Vec<t!()>;
}
