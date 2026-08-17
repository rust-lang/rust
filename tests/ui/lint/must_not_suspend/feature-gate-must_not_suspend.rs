//@ edition:2018

#[must_not_suspend = "You gotta use Umm's, ya know?"] //~ ERROR the `must_not_suspend` attribute
struct Umm {
    _i: i64
}

fn main() {
}
