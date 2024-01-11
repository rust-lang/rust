// run-rustfix

// issue #119765

fn main() -> Result<(), bool> {
    None.ok_or(|| true)?
    //~^ ERROR `?` couldn't convert the error to `bool` [E0277]
}
