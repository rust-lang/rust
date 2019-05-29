// Test that we give a custom error (E0373) for the case where a
// closure is escaping current frame, and offer a suggested code edit.
// I refrained from including the precise message here, but the
// original text as of the time of this writing is:
//
//    closure may outlive the current function, but it borrows `books`,
//    which is owned by the current function

fn foo<'a>(x: &'a i32) -> Box<dyn FnMut() + 'a> {
    let mut books = vec![1,2,3];
    Box::new(|| books.push(4))
    //~^ ERROR E0373
}

fn main() { }
