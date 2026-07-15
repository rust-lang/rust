// Ensures that we properly lint
// a removed 'expression' resulting from a macro
// in trailing expression position

macro_rules! expand_it {
    () => {
        #[cfg(false)] 25; //~ ERROR macro expansion ignores `;`
    }
}

fn main() {
    expand_it!()
}
