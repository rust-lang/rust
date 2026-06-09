// This is a reduction of a concrete test illustrating a case that was
// annoying to Rust developer stephaneyfx (see issue #46413).
//
// With resolving issue #54556, pnkfelix hopes that the new diagnostic
// output produced by NLL helps to *explain* the semantic significance
// of temp drop order, and thus why storing the result in `x` and then
// returning `x` works.

pub struct Statement;

pub struct Rows<'stmt>(&'stmt Statement);

impl<'stmt> Drop for Rows<'stmt> {
    fn drop(&mut self) {}
}

impl<'stmt> Iterator for Rows<'stmt> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn get_names() -> Option<String> {
    let stmt = Statement;
    let rows = Rows(&stmt); //~ ERROR does not live long enough
    rows.map(|row| row).next()
    // let x = rows.map(|row| row).next();
    // x
    //
    // Removing the map works too as does removing the Drop impl.
}

fn main() {}
