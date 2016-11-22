#![feature(plugin)]
#![plugin(clippy)]

fn get_number() -> usize {
    10
}

fn get_reference(n : &usize) -> &usize {
    n
}

#[allow(many_single_char_names)]
#[allow(unused_variables)]
#[deny(deref_addrof)]
fn main() {
    let a = 10;
    let aref = &a;

    let b = *&a;
    //~^ERROR immediately dereferencing a reference
    //~|HELP try this
    //~|SUGGESTION a

    let b = *&get_number();
    //~^ERROR immediately dereferencing a reference
    //~|HELP try this
    //~|SUGGESTION get_number()

    let b = *get_reference(&a);

    let bytes : Vec<usize> = vec![1, 2, 3, 4];
    let b = *&bytes[1..2][0];
    //~^ERROR immediately dereferencing a reference
    //~|HELP try this
    //~|SUGGESTION bytes[1..2][0]

    let b = *(&a);
    //~^ERROR immediately dereferencing a reference
    //~|HELP try this
    //~|SUGGESTION a

    let b = *&&a;
    //~^ERROR immediately dereferencing a reference
    //~|HELP try this
    //~|SUGGESTION &a

    let b = **&aref;
    //~^ERROR immediately dereferencing a reference
    //~|HELP try this
    //~|SUGGESTION aref

    //This produces a suggestion of 'let b = *&a;' which is still incorrect
    let b = **&&a;
    //~^ERROR immediately dereferencing a reference
    //~|HELP try this
    //~|SUGGESTION a

    {
        let mut x = 10;
        let y = *&mut x;
        //~^ERROR immediately dereferencing a reference
        //~|HELP try this
        //~|SUGGESTION x
    }

    {
        //This produces a suggestion of 'let y = *&mut x' which is still incorrect
        let mut x = 10;
        let y = **&mut &mut x;
        //~^ERROR immediately dereferencing a reference
        //~|HELP try this
        //~|SUGGESTION x
    }
}
