//@ check-pass

// Make sure that a `stmt` nonterminal does not eagerly match against
// a `pat`, since this will always cause a parse error...

macro_rules! m {
    ($pat:pat) => {};
    ($stmt:stmt) => {};
}

macro_rules! m2 {
    ($stmt:stmt) => {
        m! { $stmt }
    };
}

m2! { let x = 1 }

fn main() {}
