//@ edition:2024
// #42753
mod partial_eq {
    struct S;
    impl PartialEq<S> for S {
        fn eq(&self, _: &S) -> bool {
            true
        }
    }

    const C: S = S;
    const V: Vec<()> = vec![];

    fn foo() {
        match Some(S) {
            Some(C) => {} //~ ERROR: constant of non-structural type
            Some(C) if true => {} //~ ERROR: constant of non-structural type
            None => {}
        }
        if let Some(C) = Some(S) {} //~ ERROR: constant of non-structural type
        if let Some(C) = Some(S) && let Some(1) = Some(2) {} //~ ERROR: constant of non-structural type
        let Some(C) = Some(S) else { return; }; //~ ERROR: constant of non-structural type
        match vec![] {
            V => {} //~ ERROR: constant of non-structural type
            _ => {}
        }
        if let V = vec![] {} //~ ERROR: constant of non-structural type
        let V = vec![] else { return; }; //~ ERROR: constant of non-structural type
        let V = Vec::new() else { return; }; //~ ERROR: constant of non-structural type
    }
}

mod not_partial_eq {
    struct S;

    const C: S = S;

    fn foo() {
        match Some(S) {
            Some(C) => {} //~ ERROR: constant of non-structural type
            Some(C) if true => {} //~ ERROR: constant of non-structural type
            None => {}
        }
        if let Some(C) = Some(S) {} //~ ERROR: constant of non-structural type
        if let Some(C) = Some(S) && let Some(1) = Some(2) {} //~ ERROR: constant of non-structural type
        let Some(C) = Some(S) else { return; }; //~ ERROR: constant of non-structural type
    }
}

fn main() {}
