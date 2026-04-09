#![warn(clippy::manual_pop_if)]
#![allow(clippy::collapsible_if, clippy::redundant_closure)]
//@no-rustfix

fn main() {}

fn is_some_and_pattern(mut vec: Vec<i32>) {
    if false {
        // something
    } else if vec.last().is_some_and(|x| *x > 2) {
        vec.pop().unwrap();
    }
    //~^^^ manual_pop_if

    //~v manual_pop_if
    if vec.last().is_some_and(|x| *x > 2) {
        let val = vec.pop().unwrap();
        println!("Popped: {}", val);
    }

    //~v manual_pop_if
    if vec.last().is_some_and(|x| *x > 2) {
        println!("Popped: {}", vec.pop().unwrap());
    }

    //~v manual_pop_if
    if vec.last().is_some_and(|x| *x > 2) {
        // a comment before the pop
        vec.pop().unwrap();
    }

    //~v manual_pop_if
    if vec.last().is_some_and(|x| *x > 2) {
        vec.pop().unwrap();
        // a comment after the pop
    }
}

fn if_let_pattern(mut vec: Vec<i32>) {
    //~v manual_pop_if
    if let Some(x) = vec.last() {
        if *x > 2 {
            let val = vec.pop().unwrap();
            println!("Popped: {}", val);
        }
    }

    //~v manual_pop_if
    if let Some(x) = vec.last() {
        if *x > 2 {
            println!("Popped: {}", vec.pop().unwrap());
        }
    }

    //~v manual_pop_if
    if let Some(x) = vec.last() {
        if *x > 2 {
            // a comment before the pop
            vec.pop().unwrap();
        }
    }

    //~v manual_pop_if
    if let Some(x) = vec.last() {
        if *x > 2 {
            vec.pop().unwrap();
            // a comment after the pop
        }
    }
}

fn let_chain_pattern(mut vec: Vec<i32>) {
    //~v manual_pop_if
    if let Some(x) = vec.last()
        && *x > 2
    {
        let val = vec.pop().unwrap();
        println!("Popped: {}", val);
    }

    //~v manual_pop_if
    if let Some(x) = vec.last()
        && *x > 2
    {
        println!("Popped: {}", vec.pop().unwrap());
    }

    //~v manual_pop_if
    if let Some(x) = vec.last()
        && *x > 2
    {
        // a comment before the pop
        vec.pop().unwrap();
    }

    //~v manual_pop_if
    if let Some(x) = vec.last()
        && *x > 2
    {
        vec.pop().unwrap();
        // a comment after the pop
    }
}

fn map_unwrap_or_pattern(mut vec: Vec<i32>) {
    //~v manual_pop_if
    if vec.last().map(|x| *x > 2).unwrap_or(false) {
        let val = vec.pop().unwrap();
        println!("Popped: {}", val);
    }

    //~v manual_pop_if
    if vec.last().map(|x| *x > 2).unwrap_or(false) {
        println!("Popped: {}", vec.pop().unwrap());
    }

    //~v manual_pop_if
    if vec.last().map(|x| *x > 2).unwrap_or(false) {
        // a comment before the pop
        vec.pop().unwrap();
    }

    //~v manual_pop_if
    if vec.last().map(|x| *x > 2).unwrap_or(false) {
        vec.pop().unwrap();
        // a comment after the pop
    }
}
