enum test { thing = 3u } //! ERROR mismatched types
//!^ ERROR expected signed integer constant
fn main() {
    log(error, thing as int);
    assert(thing as int == 3);
}
