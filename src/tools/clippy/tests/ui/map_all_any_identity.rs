#![warn(clippy::map_all_any_identity)]

fn main() {
    let _ = ["foo"].into_iter().map(|s| s == "foo").any(|a| a);
    //~^ map_all_any_identity

    let _ = ["foo"].into_iter().map(|s| s == "foo").all(std::convert::identity);
    //~^ map_all_any_identity

    //
    // Do not lint
    //
    // Not identity
    let _ = ["foo"].into_iter().map(|s| s.len()).any(|n| n > 0);
    // Macro
    macro_rules! map {
        ($x:expr) => {
            $x.into_iter().map(|s| s == "foo")
        };
    }
    map!(["foo"]).any(|a| a);
}
