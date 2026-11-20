//@ check-pass
//@ edition: 2021
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ compile-flags: -Zpolonius=off
//@[current] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally

fn identity<'a, T: ?Sized>(x: &'a T) -> &'a T {
    let y = x;
    y
}

fn signature_only<'a, T: ?Sized>() {
    let ordinary = |_: &'a T| {};
    let asynchronous = async |_: &'a T| {};
    let _locals = (ordinary, asynchronous);
}

fn covariance<'a>(input: &'a str) -> Option<&'a str> {
    let mut value = Some("static");
    value = Some(input);
    value
}

fn main() {
    let mut references = Vec::new();
    {
        let text = String::from("temporary");
        references.push(identity::<str>(&text));
        let _ = covariance(&text);
    }
    // The references are no longer used, only implicitly dropped.
}
