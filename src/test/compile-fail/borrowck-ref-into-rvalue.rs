fn main() {
    let msg;
    match Some(~"Hello") { //~ ERROR illegal borrow
        Some(ref m) => {
            msg = m;
        },  
        None => { fail }
    }   
    io::println(*msg);
}

