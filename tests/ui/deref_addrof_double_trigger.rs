#[warn(clippy::deref_addrof)]
#[allow(unused_variables)]
fn main() {
    let a = 10;

    //This produces a suggestion of 'let b = *&a;' which
    //will trigger the 'clippy::deref_addrof' lint again
    let b = **&&a;

    {
        let mut x = 10;
        let y = *&mut x;
    }

    {
        //This produces a suggestion of 'let y = *&mut x' which
        //will trigger the 'clippy::deref_addrof' lint again
        let mut x = 10;
        let y = **&mut &mut x;
    }
}
