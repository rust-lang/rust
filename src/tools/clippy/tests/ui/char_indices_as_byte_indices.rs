#![feature(round_char_boundary)]
#![warn(clippy::char_indices_as_byte_indices)]

trait StrExt {
    fn use_index(&self, _: usize);
}
impl StrExt for str {
    fn use_index(&self, _: usize) {}
}

fn bad(prim: &str, string: String) {
    for (idx, _) in prim.chars().enumerate() {
        let _ = prim[..idx];
        //~^ char_indices_as_byte_indices
        prim.split_at(idx);
        //~^ char_indices_as_byte_indices

        // This won't panic, but it can still return a wrong substring
        let _ = prim[..prim.floor_char_boundary(idx)];
        //~^ char_indices_as_byte_indices

        // can't use #[expect] here because the .fixed file will still have the attribute and create an
        // unfulfilled expectation, but make sure lint level attributes work on the use expression:
        #[allow(clippy::char_indices_as_byte_indices)]
        let _ = prim[..idx];
    }

    for c in prim.chars().enumerate() {
        let _ = prim[..c.0];
        //~^ char_indices_as_byte_indices
        prim.split_at(c.0);
        //~^ char_indices_as_byte_indices
    }

    for (idx, _) in string.chars().enumerate() {
        let _ = string[..idx];
        //~^ char_indices_as_byte_indices
        string.split_at(idx);
        //~^ char_indices_as_byte_indices
    }
}

fn good(prim: &str, prim2: &str) {
    for (idx, _) in prim.chars().enumerate() {
        // Indexing into a different string
        let _ = prim2[..idx];

        // Unknown use
        std::hint::black_box(idx);

        // Method call to user defined extension trait
        prim.use_index(idx);

        // str method taking a usize that doesn't represent a byte index
        prim.splitn(idx, prim2);
    }

    let mut string = "äa".to_owned();
    for (idx, _) in string.clone().chars().enumerate() {
        // Even though the receiver is the same expression, it should not be treated as the same value.
        string.clone().remove(idx);
    }
}

fn main() {}
